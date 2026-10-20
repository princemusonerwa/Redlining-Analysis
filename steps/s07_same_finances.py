"""Step 07 - do applicants with the SAME finances get different outcomes?

Two complementary designs, both comparing each group with White applicants:

A. Regression ladder (logistic, approval = 1). Controls are added in blocks so
   the reader can watch how much of the raw gap is "explained":
     M0 group only (raw gap)
     M1 + applicant finances (income, loan size, property value, LTV, DTI,
          credit-score model, age)
     M2 + loan product (type, purpose, lien, occupancy, term, ...; year)
     M3 + place (state fixed effects, tract income level)
     M4 linear probability model with LENDER fixed effects (compares
          applicants at the same lender)
   Effects are reported as odds ratios and as an average approval-probability
   gap in percentage points for the focus group (counterfactual: same person
   predicted as White).

B. Matching: each applicant is placed in a cell of near-identical financial
   profile (same year, state, loan type, purpose, lien, occupancy, and income /
   loan-amount / LTV / DTI / credit-score bins); approval rates are compared
   only within shared cells.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import time

import numpy as np
import pandas as pd
import statsmodels.api as sm

import config
from common import (FINANCE_CONTROLS, LOAN_CONTROLS, PLACE_CONTROLS, control_frame, dummies,
                    get_logger, load_prepared, mark_done, plt, save_json, savefig, step_dir)

NAME = "s07_same_finances"
GROUPS = ["Black", "Hispanic", "Asian", "Native American", "Pacific Islander"]


def group_dummies(g):
    return pd.DataFrame({f"grp_{x}": (g == x).astype("float32") for x in GROUPS + ["Joint", "Multiracial", "Unknown"]})


def independent_columns(X, keep_first=("grp_", "band_"), tol=1e-9):
    """Drop columns that are constant or linear combinations of earlier ones.
    Columns with a keep_first prefix are tested first so they are never the ones removed."""
    cols = sorted(X.columns, key=lambda c: 0 if c.startswith(keep_first) else 1)
    A = X[cols].to_numpy(dtype=np.float64)
    A = np.column_stack([np.ones(len(A)), A])          # intercept goes first
    G = A.T @ A
    kept = [0]
    for j in range(1, G.shape[0]):
        K = np.array(kept)
        b = np.linalg.lstsq(G[np.ix_(K, K)], G[K, j], rcond=None)[0]
        resid = G[j, j] - G[K, j] @ b
        if resid > tol * max(G[j, j], 1.0):
            kept.append(j)
    return [cols[j - 1] for j in kept[1:]]


def fit_logit(y, X, log, label, clusters):
    """Logit with county-clustered SEs. Control dummies whose rows are all
    approved (or all denied) are dropped first: they cause perfect separation
    and stop the optimiser from converging. Same for levels with fewer than 10
    approvals or denials (near-separation makes the Hessian singular)."""
    yv = np.asarray(y)
    def near_separating(c):
        yy = yv[X[c].to_numpy() == 1]
        return min(yy.sum(), len(yy) - yy.sum()) < 10   # fewer than 10 approvals or denials
    bad = [c for c in X.columns if not c.startswith(("grp_", "band_")) and near_separating(c)]
    X = X.drop(columns=bad)
    ok = independent_columns(X)
    if len(ok) < X.shape[1]:
        log.info("%s: dropped %d collinear columns", label, X.shape[1] - len(ok))
    X = sm.add_constant(X[ok], has_constant="add")
    kw = dict(disp=0, cov_type="cluster", cov_kwds={"groups": clusters})
    res = sm.Logit(y, X).fit(method="newton", maxiter=100, **kw)
    if not res.mle_retvals.get("converged", True):
        log.info("%s: newton did not converge, refining with lbfgs", label)
        res = sm.Logit(y, X).fit(method="lbfgs", maxiter=3000, start_params=res.params, **kw)
    if bad:
        log.info("%s: dropped %d perfectly-predicting dummies", label, len(bad))
    return res, X


def counterfactual_gap(res, X, g, focus, coef=None):
    """Mean predicted approval for focus-group rows minus the same rows predicted as White.
    `coef` overrides the group coefficient (used to turn its CI bounds into a gap range)."""
    rows = X[g == focus]
    params = res.params.copy()
    if coef is not None:
        params[f"grp_{focus}"] = coef
    pred = lambda Z: 1 / (1 + np.exp(-(Z.to_numpy() @ params.to_numpy())))
    rows_w = rows.copy()
    rows_w[[c for c in rows.columns if c.startswith("grp_")]] = 0.0
    return float(np.mean(pred(rows) - pred(rows_w)))


def lender_fe_lpm(y, X, lender):
    """Linear probability model with lender fixed effects via within-lender demeaning."""
    d = X.copy()
    d["__y"] = y
    gm = d.groupby(lender.to_numpy()).transform("mean")
    dm = d - gm
    keep = dm.columns[(dm.abs().sum() > 0)]
    dm = dm[keep]
    res = sm.OLS(dm["__y"], dm.drop(columns="__y")).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(lender)[0]})
    return res


def matched_gap(df, cell, focus, ref="White", n_boot=config.N_BOOT, seed=config.SEED):
    d = pd.DataFrame({"cell": cell, "g": df["group"].astype(str), "y": df["approved"]})
    d = d[d.g.isin([focus, ref])]
    t = d.groupby(["cell", "g"]).y.agg(["sum", "count"]).unstack("g")
    t = t.dropna()
    t = t[(t[("count", focus)] > 0) & (t[("count", ref)] > 0)]
    if t.empty:
        return {"gap_pp": np.nan, "lo": np.nan, "hi": np.nan, "matched_focus": 0,
                "focus_total": int((d.g == focus).sum()), "matched_share": 0.0, "cells": 0}
    nf = t[("count", focus)].to_numpy()
    diff = (t[("sum", focus)] / t[("count", focus)] - t[("sum", ref)] / t[("count", ref)]).to_numpy()
    est = float(np.sum(nf * diff) / nf.sum())
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(nf), len(nf))
        boots.append(np.sum(nf[i] * diff[i]) / nf[i].sum())
    total_focus = int((d.g == focus).sum())
    return {"gap_pp": est, "lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975)),
            "matched_focus": int(nf.sum()), "focus_total": total_focus,
            "matched_share": float(nf.sum() / total_focus) if total_focus else np.nan, "cells": len(nf)}


def run():
    t0 = time.time()
    log = get_logger(NAME)
    out = step_dir(NAME)
    df = load_prepared()
    y = df["approved"].astype(float)
    g = df["group"].astype(str)
    ctrl = control_frame(df)
    clusters = pd.factorize(df["county_code"].fillna("NA"))[0]

    blocks = {
        "M0 raw gap": [],
        "M1 + applicant finances": FINANCE_CONTROLS,
        "M2 + loan product & year": FINANCE_CONTROLS + LOAN_CONTROLS,
        "M3 + state & tract income": FINANCE_CONTROLS + LOAN_CONTROLS + PLACE_CONTROLS,
    }
    G = group_dummies(g)
    rows = []
    for label, cols in blocks.items():
        tm = time.time()
        X = pd.concat([G, dummies(ctrl, cols)], axis=1)
        res, Xc = fit_logit(y, X, log, label, clusters)
        ci = np.exp(res.conf_int())
        for focus in GROUPS:
            k = f"grp_{focus}"
            if k not in res.params:
                continue
            rows.append({"model": label, "group": focus, "odds_ratio": float(np.exp(res.params[k])),
                         "or_lo": float(ci.loc[k, 0]), "or_hi": float(ci.loc[k, 1]),
                         "p_value": float(res.pvalues[k]),
                         "gap_pp": counterfactual_gap(res, Xc, g, focus),
                         "gap_lo": counterfactual_gap(res, Xc, g, focus, res.conf_int().loc[k, 0]),
                         "gap_hi": counterfactual_gap(res, Xc, g, focus, res.conf_int().loc[k, 1]),
                         "n": int(res.nobs), "pseudo_r2": float(res.prsquared)})
        log.info("%-28s Black OR %.3f  gap %.2f pp  (%d params, %.0fs)", label,
                 [r for r in rows if r["model"] == label and r["group"] == "Black"][0]["odds_ratio"],
                 100 * [r for r in rows if r["model"] == label and r["group"] == "Black"][0]["gap_pp"],
                 X.shape[1], time.time() - tm)

    # lender fixed effects (linear probability model)
    tm = time.time()
    X = pd.concat([G, dummies(ctrl, FINANCE_CONTROLS + LOAN_CONTROLS + PLACE_CONTROLS)], axis=1)
    lpm = lender_fe_lpm(y, X, df["lei"].astype(str))
    ci = lpm.conf_int()
    for focus in GROUPS:
        k = f"grp_{focus}"
        if k in lpm.params:
            rows.append({"model": "M4 + lender fixed effects (LPM)", "group": focus, "odds_ratio": np.nan,
                         "or_lo": np.nan, "or_hi": np.nan, "p_value": float(lpm.pvalues[k]),
                         "gap_pp": float(lpm.params[k]), "gap_lo": float(ci.loc[k, 0]),
                         "gap_hi": float(ci.loc[k, 1]), "n": int(lpm.nobs), "pseudo_r2": np.nan})
    log.info("M4 lender FE done (%.0fs)", time.time() - tm)
    reg = pd.DataFrame(rows)
    reg.to_csv(out / "regression_ladder.csv", index=False)

    # ---- matching ---------------------------------------------------------
    strict = ctrl[["activity_year", "state_code", "loan_type", "loan_purpose", "lien_status",
                   "occupancy_type", "income_bin", "loan_amount_bin", "ltv_bin", "dti_bin",
                   "applicant_credit_score_type"]].agg("|".join, axis=1)
    broad = ctrl[["activity_year", "loan_type", "loan_purpose", "lien_status", "occupancy_type",
                  "income_bin", "loan_amount_bin", "dti_bin"]].agg("|".join, axis=1)
    mrows = []
    for label, cell in [("strict match", strict), ("broad match", broad)]:
        for focus in ["Black", "Hispanic", "Asian"]:
            r = matched_gap(df, cell, focus)
            r.update(design=label, group=focus)
            mrows.append(r)
    match = pd.DataFrame(mrows)
    match.to_csv(out / "matching.csv", index=False)
    log.info("Matching: %s", match[["design", "group", "gap_pp", "matched_share"]].to_string(index=False))

    # ---- figure: how the gap shrinks as controls are added -----------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, focus in zip(axes, ["Black", "Hispanic", "Asian"]):
        r = reg[reg.group == focus]
        m = match[match.group == focus]
        labels = list(r.model) + [f"Matching ({d})" for d in m.design]
        vals = list(100 * r.gap_pp) + list(100 * m.gap_pp)
        err_lo = list(100 * (r.gap_pp - r.gap_lo)) + list(100 * (m.gap_pp - m.lo))
        err_hi = list(100 * (r.gap_hi - r.gap_pp)) + list(100 * (m.hi - m.gap_pp))
        ax.barh(labels, vals, xerr=np.maximum(0, [err_lo, err_hi]), color=["#95a5a6"] + ["#2c7fb8"] * (len(r) - 1) + ["#8e44ad"] * len(m), capsize=3)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"{focus} vs White")
        ax.set_xlabel("Approval gap, percentage points")
        ax.invert_yaxis()
    fig.suptitle("Approval gap after accounting for finances (negative = less likely to be approved)")
    savefig(fig, out / "fig_gap_ladder.png")

    pick = lambda m, gr: reg[(reg.model.str.startswith(m)) & (reg.group == gr)].iloc[0].to_dict()
    save_json({"raw_black": pick("M0", "Black"), "full_black": pick("M3", "Black"),
               "lender_fe_black": pick("M4", "Black"), "matching": match.to_dict("records"),
               "ladder": reg.to_dict("records")}, out / "summary.json")
    mark_done(NAME, t0)


if __name__ == "__main__":
    run()
