"""Step 04 - prepare the cleaned data for downstream analyses.

Keeps action codes 1/2/3, builds consistent demographic groups, encodes
categories, and saves the Parquet contract consumed by subsequent steps.
Missing numeric values stay missing; there is no whole-sample imputation or
percentile capping. The field audit is in-sample descriptive, not validated AUC.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config
from common import (GROUP_ORDER, get_logger, mark_done, save_json, savefig,
                    step_dir, bootstrap_ci, plt)

NAME = "s04_prepare"

RAW_COLS = (["activity_year", "lei", "state_code", "county_code", "census_tract",
             "derived_race", "derived_ethnicity", "derived_sex", "action_taken",
             "tract_minority_population_percent"]
            + config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES + config.LEAKY_FIELDS
            + ["derived_loan_product_type"])
RAW_COLS = list(dict.fromkeys(RAW_COLS))

DTI_RANGES = {"<20%": 15.0, "20%-<30%": 25.0, "30%-<36%": 33.0, "50%-60%": 55.0, ">60%": 65.0}


def parse_dti(s):
    mapped = s.map(DTI_RANGES)
    return mapped.fillna(pd.to_numeric(s, errors="coerce"))


from common import make_group


def single_feature_auc(x, y):
    """In-sample descriptive AUC (not validation performance), scoring each row with the approval rate of its bin.
    Missing values get their own bin, so this also captures missingness leakage."""
    if pd.api.types.is_numeric_dtype(x):
        bins = pd.qcut(x, 20, duplicates="drop").astype(str)
        bins = bins.where(x.notna(), "MISSING")
    else:
        bins = x.astype(str)
    score = y.groupby(bins).transform("mean")
    return roc_auc_score(y, score)


def run():
    t0 = time.time()
    log = get_logger(NAME)
    out = step_dir(NAME)

    log.info("Reading raw file %s", config.CLEAN_CSV.name)
    df = pd.read_csv(config.CLEAN_CSV, usecols=RAW_COLS, dtype=str, low_memory=False)
    n_raw = len(df)
    if not df["action_taken"].isin(["1", "2", "3"]).any():
        raise ValueError("No decided applications in the cleaned data")
    action_counts = df["action_taken"].value_counts().sort_index().to_dict()

    keep = df["action_taken"].isin(config.APPROVED_CODES | config.DENIED_CODES)
    df = df[keep].copy()
    df["approved"] = df["action_taken"].isin(config.APPROVED_CODES).astype("int8")
    log.info("Rows: raw %d -> decided applications %d", n_raw, len(df))

    # ---- numeric parsing ------------------------------------------------
    df["debt_to_income_ratio"] = parse_dti(df["debt_to_income_ratio"])
    num = [c for c in config.NUMERIC_FEATURES if c != "debt_to_income_ratio"] + \
          ["interest_rate", "rate_spread", "total_loan_costs", "origination_charges",
           "tract_minority_population_percent"]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Cleaning already applied fixed validity rules. No global percentile caps.
    caps = {}

    # ---- categoricals -----------------------------------------------------
    for c in config.CATEGORICAL_FEATURES + ["purchaser_type", "hoepa_status",
                                            "initially_payable_to_institution",
                                            "denial_reason-1", "derived_loan_product_type"]:
        s = df[c].fillna("NA").replace({"1111": "Exempt", "8888": "NA"})
        df[c] = s.astype("category")

    df["group"] = make_group(df["derived_race"], df["derived_ethnicity"])
    df["county_code"] = df["county_code"].str.replace(r"\.0$", "", regex=True).str.zfill(5)
    df["census_tract"] = df["census_tract"].str.replace(r"\.0$", "", regex=True).str.zfill(11)
    df["activity_year"] = df["activity_year"].astype("category")
    for c in ["derived_race", "derived_ethnicity", "derived_sex", "state_code", "lei", "action_taken"]:
        df[c] = df[c].astype("category")
    df = df.reset_index(drop=True)

    # ---- leakage audit ----------------------------------------------------
    log.info("Leakage audit")
    y = df["approved"]
    rows = []
    for c in config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES + config.LEAKY_FIELDS:
        x = df[c]
        miss = x.isna() if pd.api.types.is_numeric_dtype(x) else x.isin(["NA"])
        rows.append({
            "feature": c,
            "role": "EXCLUDED (post-decision)" if c in config.LEAKY_FIELDS else "model input",
            "missing_if_approved": miss[y == 1].mean(),
            "missing_if_denied": miss[y == 0].mean(),
            "auc_alone": single_feature_auc(x, y),
        })
    audit = pd.DataFrame(rows).sort_values("auc_alone", ascending=False)
    audit.to_csv(out / "leakage_audit.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 7))
    a = audit.sort_values("auc_alone")
    colors = ["#c0392b" if r.startswith("EXCL") else "#2c7fb8" for r in a.role]
    ax.barh(a.feature, a.auc_alone, color=colors)
    ax.axvline(0.5, color="grey", lw=1)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("AUC from this one field alone (0.5 = no information)")
    ax.set_title("Leakage audit: red fields are only known after the decision and are excluded")
    savefig(fig, out / "fig_leakage_audit.png")

    # ---- descriptive approval rates --------------------------------------
    def rate_ci(s):
        lo, hi = bootstrap_ci(lambda i: {"r": s.values[i].mean()}, len(s), n_boot=200)
        return s.mean(), lo["r"], hi["r"]

    by_group = []
    for g, s in df.groupby("group", observed=True)["approved"]:
        m, lo, hi = rate_ci(s)
        by_group.append({"group": g, "n": len(s), "approval_rate": m, "ci_low": lo, "ci_high": hi})
    by_group = pd.DataFrame(by_group)
    by_group.to_csv(out / "approval_by_group.csv", index=False)

    df["income_quartile"] = pd.qcut(df["income"], 4, duplicates="drop")
    by_inc = (df[df.group.isin([config.REFERENCE_GROUP] + config.FOCUS_GROUPS)]
              .groupby(["income_quartile", "group"], observed=True)["approved"].mean().unstack())
    by_inc.to_csv(out / "approval_by_income_quartile.csv")
    by_year = df.groupby(["activity_year", "group"], observed=True)["approved"].mean().unstack()
    by_year.to_csv(out / "approval_by_year.csv")
    df = df.drop(columns="income_quartile")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    bg = by_group.set_index("group").loc[[g for g in GROUP_ORDER if g in set(by_group.group)]]
    axes[0].bar(bg.index, bg.approval_rate, yerr=np.maximum(0, [bg.approval_rate - bg.ci_low, bg.ci_high - bg.approval_rate]),
                color="#2c7fb8", capsize=3)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_title("Actual approval rate by group (95% CI)")
    axes[0].tick_params(axis="x", rotation=40)
    by_inc.plot(kind="bar", ax=axes[1], rot=0)
    axes[1].set_ylim(0.5, 1.0)
    axes[1].set_title("Approval rate by income quartile")
    axes[1].set_xlabel("")
    savefig(fig, out / "fig_approval_by_group.png")

    df.to_parquet(out / "prepared.parquet", index=False)
    summary = {
        "n_raw": n_raw, "years": sorted(df["activity_year"].astype(str).unique().tolist()), "n_input_columns": len(RAW_COLS), "action_taken_counts_raw": action_counts,
        "n_analysis": len(df), "approval_rate": float(y.mean()),
        "group_counts": df["group"].value_counts().to_dict(),
        "caps": caps, "states": int(df["state_code"].nunique()),
        "top_leaky": audit.head(8).to_dict("records"),
    }
    save_json(summary, out / "summary.json")
    log.info("Saved %d rows; approval rate %.3f", len(df), y.mean())
    mark_done(NAME, t0)


if __name__ == "__main__":
    run()
