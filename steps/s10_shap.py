"""Step 10 - SHAP explanations for the best model.

SHAP splits each prediction (on the model output scale) into additive pushes from each input.
Outputs:
  * global importance (mean |SHAP|) and beeswarm, on a random test sample
  * dependence plots for the top numeric drivers
  * gap decomposition: because SHAP is additive, the average predicted
    log-odds gap between Black and White applicants equals the sum over
    features of their average SHAP differences. This shows WHICH inputs carry
    the gap, even though race itself is not an input.
  * two individual explanations (waterfall charts)
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import time

import joblib
import numpy as np
import pandas as pd
import shap

import config
from common import get_logger, load_prepared, mark_done, plt, save_json, savefig, step_dir
from s05_models import tree_matrix

NAME = "s10_shap"
TREE_MODELS = ["LightGBM", "XGBoost", "CatBoost", "Random Forest"]


def positive_class(sv):
    if isinstance(sv, list):
        return sv[1]
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv


def run():
    t0 = time.time()
    log = get_logger(NAME)
    out = step_dir(NAME)
    test = pd.read_csv(config.OUT / "s05_models" / "test_metrics.csv", index_col=0)
    cv = pd.read_csv(config.OUT / "s05_models" / "cv_summary.csv", index_col=0).sort_values("roc_auc_mean", ascending=False)
    trees = [m for m in cv.index if m in TREE_MODELS]
    best = trees[0]
    # Random Forest SHAP is slow; prefer a boosted model if it is within 0.005 AUC
    if best == "Random Forest":
        alt = [m for m in trees if m != "Random Forest"][0]
        if cv.loc[best, "roc_auc_mean"] - cv.loc[alt, "roc_auc_mean"] < 0.005:
            best = alt
    log.info("Explaining %s (test AUC %.4f)", best, test.loc[best, "roc_auc"])
    unit = "probability" if best == "Random Forest" else "log-odds"
    model = joblib.load(config.OUT / "s05_models" / f"model_{best.replace(' ', '_')}.joblib")

    df = load_prepared()
    idx_te = np.load(config.OUT / "s05_models" / "test_idx.npy")
    X = tree_matrix(df).iloc[idx_te].reset_index(drop=True)
    grp = df["group"].astype(str).to_numpy()[idx_te]
    rng = np.random.default_rng(config.SEED)

    explainer = shap.TreeExplainer(model)

    # ---- global sample -----------------------------------------------------
    n_glob = min(config.SHAP_ROWS // 2, len(X))
    gi = rng.choice(len(X), n_glob, replace=False)
    sv = positive_class(explainer.shap_values(X.iloc[gi]))
    imp = pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False)
    imp.to_csv(out / "global_importance.csv", header=["mean_abs_shap"])
    log.info("Top features: %s", ", ".join(imp.index[:6]))

    fig, ax = plt.subplots(figsize=(7, 5.5))
    top = imp.head(15)[::-1]
    ax.barh(top.index, top.values, color="#2c7fb8")
    ax.set_xlabel(f"Mean |SHAP| (average push on approval {unit})")
    ax.set_title(f"What drives the {best} model's approvals")
    savefig(fig, out / "fig_shap_importance.png")

    plt.figure()
    shap.summary_plot(sv, X.iloc[gi], max_display=15, show=False, plot_size=(8, 6))
    plt.title("Each dot is one applicant: red = high value, blue = low value")
    savefig(plt.gcf(), out / "fig_shap_beeswarm.png")

    num_top = [c for c in imp.index if c in config.NUMERIC_FEATURES][:4]
    fig, axes = plt.subplots(1, len(num_top), figsize=(4 * len(num_top), 3.4))
    for ax, c in zip(np.atleast_1d(axes), num_top):
        j = list(X.columns).index(c)
        ax.scatter(X.iloc[gi][c], sv[:, j], s=3, alpha=0.3)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xlabel(c)
        ax.set_ylabel("SHAP")
        lo, hi = X.iloc[gi][c].quantile([0.01, 0.99])
        ax.set_xlim(lo, hi)
    fig.suptitle("How the push changes with the value (dependence plots)")
    savefig(fig, out / "fig_shap_dependence.png")

    # ---- gap decomposition: Black vs White, Hispanic vs White ----------------------
    per = config.SHAP_ROWS // 4
    decomp = {}
    for focus in ["Black", "Hispanic"]:
        fi = np.where(grp == focus)[0]
        wi = np.where(grp == config.REFERENCE_GROUP)[0]
        fi = rng.choice(fi, min(per, len(fi)), replace=False)
        wi = rng.choice(wi, min(per, len(wi)), replace=False)
        sf = positive_class(explainer.shap_values(X.iloc[fi])).mean(0)
        sw = positive_class(explainer.shap_values(X.iloc[wi])).mean(0)
        diff = pd.Series(sf - sw, index=X.columns).sort_values()
        decomp[focus] = diff
    dec = pd.DataFrame(decomp)
    dec.to_csv(out / "gap_decomposition.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, focus in zip(axes, ["Black", "Hispanic"]):
        s = dec[focus].reindex(dec[focus].abs().sort_values(ascending=False).index[:12])[::-1]
        ax.barh(s.index, s.values, color=["#c0392b" if v < 0 else "#27ae60" for v in s.values])
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"Where the predicted {focus}-White gap comes from\n(total {dec[focus].sum():+.3f} {unit})")
        ax.set_xlabel("Average SHAP difference (red = lowers approval for this group)")
    savefig(fig, out / "fig_shap_gap_decomposition.png")

    # ---- two individual explanations -----------------------------------------
    p = model.predict_proba(X.iloc[gi])[:, 1]
    ex_ids = {"likely denied": gi[int(np.argmin(np.abs(p - 0.25)))],
              "likely approved": gi[int(np.argmin(np.abs(p - 0.9)))]}
    base = explainer.expected_value
    base = float(base[1] if np.ndim(base) and np.size(base) > 1 else np.ravel([base])[0])
    for label, i in ex_ids.items():
        v = positive_class(explainer.shap_values(X.iloc[[i]]))[0]
        e = shap.Explanation(values=v, base_values=base, data=X.iloc[i].values, feature_names=list(X.columns))
        plt.figure()
        shap.plots.waterfall(e, max_display=10, show=False)
        plt.title(f"One applicant, {label}: how each input moved the prediction ({unit})")
        savefig(plt.gcf(), out / f"fig_shap_waterfall_{label.split()[1]}.png")

    save_json({"model": best, "top_features": imp.head(15).to_dict(),
               "gap_decomposition_top": {f: dec[f].reindex(dec[f].abs().sort_values(ascending=False).index[:8]).to_dict()
                                         for f in dec.columns},
               "gap_total_logodds": dec.sum().to_dict(), "output_unit": unit}, out / "summary.json")
    mark_done(NAME, t0, {"model": best})


if __name__ == "__main__":
    run()
