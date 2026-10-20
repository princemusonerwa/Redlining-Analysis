"""Step 05 - train and compare six approval models without leaking fields.

* race / ethnicity / sex / tract minority share are NOT model inputs
* 5-fold stratified cross-validation (on a stratified subsample of the
  training set, for runtime) + a final fit on the full training set,
  evaluated on an untouched 20% test set
* leakage demonstration: the same LightGBM with and without interest_rate /
  rate_spread, plus a model that only sees which fields are missing
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

import config
from common import get_logger, load_prepared, mark_done, plt, save_json, savefig, step_dir

NAME = "s05_models"
SKEWED = ["loan_amount", "income", "property_value", "tract_population",
          "tract_owner_occupied_units", "tract_one_to_four_family_homes"]


def tree_matrix(df):
    """Numeric columns as-is (NaN kept); categoricals as integer codes."""
    X = df[config.NUMERIC_FEATURES].astype("float32").copy()
    for c in config.CATEGORICAL_FEATURES:
        X[c] = df[c].cat.codes.astype("int16")
    return X


def signed_log1p(x):
    """Log compression that retains zero/negative income without invalid logs."""
    return np.sign(x) * np.log1p(np.abs(x))


def linear_pipeline(model):
    num_skew = [c for c in config.NUMERIC_FEATURES if c in SKEWED]
    num_other = [c for c in config.NUMERIC_FEATURES if c not in SKEWED]
    pre = ColumnTransformer([
        ("skew", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                           ("log", FunctionTransformer(signed_log1p)),
                           ("sc", StandardScaler())]), num_skew),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                          ("sc", StandardScaler())]), num_other),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=200), config.CATEGORICAL_FEATURES),
    ])
    return Pipeline([("pre", pre), ("clf", model)])


def linear_frame(df):
    X = df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES].copy()
    for c in config.CATEGORICAL_FEATURES:
        X[c] = X[c].astype(str)
    return X


def model_specs():
    s = config.SEED
    small = config.SMOKE
    return {
        "Logistic Regression": ("linear", lambda: linear_pipeline(LogisticRegression(max_iter=3000, C=1.0))),
        "KNN": ("linear", lambda: linear_pipeline(KNeighborsClassifier(n_neighbors=50, n_jobs=2 if small else -1))),
        "Random Forest": ("tree", lambda: RandomForestClassifier(
            n_estimators=15 if small else 200, max_depth=5 if small else 20, min_samples_leaf=20, max_features="sqrt",
            n_jobs=2 if small else -1, random_state=s)),
        "XGBoost": ("tree", lambda: XGBClassifier(
            n_estimators=15 if small else 600, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", n_jobs=2 if small else -1, random_state=s, eval_metric="logloss")),
        "LightGBM": ("tree", lambda: LGBMClassifier(
            n_estimators=15 if small else 800, learning_rate=0.05, num_leaves=127, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, n_jobs=2 if small else -1, random_state=s, verbose=-1)),
        "CatBoost": ("tree", lambda: CatBoostClassifier(
            iterations=15 if small else 800, depth=8, learning_rate=0.08, thread_count=2 if small else -1, random_seed=s, verbose=0, allow_writing_files=False)),
    }


def metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc_denied": average_precision_score(1 - y, 1 - p),
        "accuracy": accuracy_score(y, yhat),
        "balanced_accuracy": balanced_accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat),
        "f1": f1_score(y, yhat),
        "denial_recall": recall_score(1 - y, 1 - yhat),
        "brier": brier_score_loss(y, p),
    }


def run():
    t0 = time.time()
    log = get_logger(NAME)
    out = step_dir(NAME)
    df = load_prepared()
    y = df["approved"].to_numpy()

    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=config.TEST_SIZE,
                                      stratify=y, random_state=config.SEED)
    np.save(out / "train_idx.npy", idx_tr)
    np.save(out / "test_idx.npy", idx_te)
    Xt, Xl = tree_matrix(df), linear_frame(df)
    log.info("Train %d / test %d; base approval rate %.3f", len(idx_tr), len(idx_te), y.mean())

    # cross-validation subsample of the training set
    rng = np.random.default_rng(config.SEED)
    cv_idx = idx_tr if len(idx_tr) <= config.CV_MAX_ROWS else \
        train_test_split(idx_tr, train_size=config.CV_MAX_ROWS, stratify=y[idx_tr], random_state=config.SEED)[0]
    skf = StratifiedKFold(config.N_FOLDS, shuffle=True, random_state=config.SEED)

    cv_rows, test_rows = [], []
    preds = pd.DataFrame({"row": idx_te, "approved": y[idx_te], "group": df["group"].to_numpy()[idx_te]})
    for name, (kind, make) in model_specs().items():
        X = Xt if kind == "tree" else Xl
        tm = time.time()
        sub = cv_idx
        if name == "KNN" and len(sub) > config.KNN_MAX_ROWS:
            sub = train_test_split(sub, train_size=config.KNN_MAX_ROWS, stratify=y[sub],
                                   random_state=config.SEED)[0]
        for k, (a, b) in enumerate(skf.split(sub, y[sub])):
            m = make()
            m.fit(X.iloc[sub[a]], y[sub[a]])
            r = metrics(y[sub[b]], m.predict_proba(X.iloc[sub[b]])[:, 1])
            r.update(model=name, fold=k)
            cv_rows.append(r)
        log.info("%-20s CV AUC %.4f (%.0fs)", name,
                 np.mean([r["roc_auc"] for r in cv_rows if r["model"] == name]), time.time() - tm)

        fit_idx = idx_tr
        if name == "KNN" and len(fit_idx) > config.KNN_MAX_ROWS:
            fit_idx = train_test_split(idx_tr, train_size=config.KNN_MAX_ROWS, stratify=y[idx_tr],
                                       random_state=config.SEED)[0]
        tm = time.time()
        m = make()
        m.fit(X.iloc[fit_idx], y[fit_idx])
        p = m.predict_proba(X.iloc[idx_te])[:, 1]
        preds[name] = p.astype("float32")
        r = metrics(y[idx_te], p)
        r.update(model=name, train_rows=len(fit_idx))
        test_rows.append(r)
        log.info("%-20s TEST AUC %.4f acc %.4f (final fit %.0fs)", name, r["roc_auc"], r["accuracy"], time.time() - tm)
        if name != "KNN":
            joblib.dump(m, out / f"model_{name.replace(' ', '_')}.joblib", compress=3)

    cv = pd.DataFrame(cv_rows)
    cv_summary = cv.drop(columns="fold").groupby("model").agg(["mean", "std"])
    cv_summary.columns = [f"{a}_{b}" for a, b in cv_summary.columns]
    cv.to_csv(out / "cv_folds.csv", index=False)
    cv_summary.to_csv(out / "cv_summary.csv")
    test = pd.DataFrame(test_rows).set_index("model").sort_values("roc_auc", ascending=False)
    test.to_csv(out / "test_metrics.csv")
    preds.to_parquet(out / "test_predictions.parquet", index=False)

    # ---- leakage demonstration (LightGBM, same split) --------------------
    log.info("Leakage demonstration")
    lgb = model_specs()["LightGBM"][1]
    leak = {}
    for label, extra in [("Without leaking fields (this study)", []),
                         ("With interest_rate + rate_spread (original setup)", ["interest_rate", "rate_spread"])]:
        X = Xt.copy()
        for c in extra:
            X[c] = df[c].astype("float32")
        m = lgb().fit(X.iloc[idx_tr], y[idx_tr])
        leak[label] = roc_auc_score(y[idx_te], m.predict_proba(X.iloc[idx_te])[:, 1])
    miss = pd.DataFrame({c: df[c].isna().astype("int8") for c in config.NUMERIC_FEATURES + ["interest_rate", "rate_spread"]})
    m = lgb().fit(miss.iloc[idx_tr], y[idx_tr])
    leak["Only 'is this field missing?' flags"] = roc_auc_score(y[idx_te], m.predict_proba(miss.iloc[idx_te])[:, 1])
    save_json(leak, out / "leakage_demo.json")

    # ---- figures ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in test.index:
        fpr, tpr, _ = roc_curve(preds.approved, preds[name])
        ax.plot(fpr, tpr, label=f"{name} (AUC {test.loc[name, 'roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on the held-out test set")
    ax.legend(fontsize=8)
    savefig(fig, out / "fig_roc.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    cs = cv_summary.loc[test.index]
    xs = np.arange(len(cs))
    for j, met in enumerate(["roc_auc", "balanced_accuracy", "f1"]):
        ax.bar(xs + (j - 1) * 0.27, cs[f"{met}_mean"], 0.27, yerr=cs[f"{met}_std"], capsize=2, label=met)
    ax.set_xticks(xs, cs.index, rotation=20)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Cross-validated performance (mean +/- sd over 5 folds)")
    ax.legend()
    savefig(fig, out / "fig_cv_metrics.png")

    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    ax.barh(list(leak.keys()), list(leak.values()), color=["#2c7fb8", "#c0392b", "#e67e22"])
    for i, v in enumerate(leak.values()):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center")
    ax.set_xlim(0.5, 1.05)
    ax.set_xlabel("Test ROC-AUC")
    ax.set_title("Effect of including potentially leaking fields")
    savefig(fig, out / "fig_leakage_demo.png")

    best = cv_summary["roc_auc_mean"].idxmax()
    save_json({"best_model": best, "test_metrics": test.reset_index().to_dict("records"),
               "majority_class_accuracy": float(max(y[idx_te].mean(), 1 - y[idx_te].mean())),
               "leakage_demo": leak, "n_train": len(idx_tr), "n_test": len(idx_te),
               "cv_rows": len(cv_idx), "model_selection": "highest training CV mean ROC-AUC", "smoke": config.SMOKE}, out / "summary.json")
    mark_done(NAME, t0, {"best_model": best})


if __name__ == "__main__":
    run()
