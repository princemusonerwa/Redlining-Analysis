"""Step 08 - neighbourhood (census-tract) test: a modern redlining check.

Question: are applicants in high-minority neighbourhoods less likely to be
approved than SIMILAR applicants in low-minority neighbourhoods, even after
accounting for their own race, finances, loan product, state and the tract's
income level? This mirrors the FFIEC redlining indicator (approval / denial
differences between high- and low-minority areas).

Also writes a tract-level table (approval rate, minority share, income) that a
later step can join to the 1930s HOLC redlining maps.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import time

import numpy as np
import pandas as pd
import statsmodels.api as sm

import config
from common import (FINANCE_CONTROLS, LOAN_CONTROLS, control_frame, dummies, get_logger,
                    load_prepared, mark_done, plt, save_json, savefig, step_dir)
from s07_same_finances import GROUPS, fit_logit, group_dummies

NAME = "s08_neighbourhood"
BANDS = [-0.01, 10, 30, 50, 80, 100.01]
BAND_LABELS = ["<10%", "10-30%", "30-50%", "50-80%", "80%+"]


def run():
    t0 = time.time()
    log = get_logger(NAME)
    out = step_dir(NAME)
    df = load_prepared()
    df = df[df["tract_minority_population_percent"].notna()].reset_index(drop=True)
    df["band"] = pd.cut(df["tract_minority_population_percent"], BANDS, labels=BAND_LABELS, right=False).astype(str)
    y = df["approved"].astype(float)
    g = df["group"].astype(str)

    raw = df.groupby("band").approved.agg(["mean", "count"]).reindex(BAND_LABELS)
    raw.to_csv(out / "approval_by_minority_band.csv")
    within = (df[g.isin(["White", "Black", "Hispanic"])]
              .groupby(["band", "group"], observed=True).approved.mean().unstack().reindex(BAND_LABELS))
    within.to_csv(out / "approval_by_band_and_group.csv")
    ctrl = control_frame(df)
    grid = df.assign(tract_income=ctrl["tract_income_bin"]).pivot_table(
        index="band", columns="tract_income", values="approved", aggfunc="mean").reindex(BAND_LABELS)
    grid.to_csv(out / "approval_by_band_and_tract_income.csv")

    # ---- regression -----------------------------------------------------------
    B = pd.get_dummies(pd.Categorical(df["band"], categories=BAND_LABELS), prefix="band",
                       drop_first=True, dtype="float32")
    B.index = df.index
    specs = {
        "raw (band only)": pd.concat([B], axis=1),
        "+ own race": pd.concat([B, group_dummies(g)], axis=1),
        "+ finances, loan, state, tract income": pd.concat(
            [B, group_dummies(g), dummies(ctrl, FINANCE_CONTROLS + LOAN_CONTROLS + ["state_code", "tract_income_bin"])], axis=1),
    }
    clusters = pd.factorize(df["county_code"].fillna("NA"))[0]
    rows = []
    for label, X in specs.items():
        tm = time.time()
        res, Xc = fit_logit(y, X, log, label, clusters)
        ci = np.exp(res.conf_int())
        for b in BAND_LABELS[1:]:
            k = f"band_{b}"
            rows_b = Xc[Xc[k] == 1]
            cf = rows_b.copy()
            cf[[c for c in Xc.columns if c.startswith("band_")]] = 0.0
            rows.append({"spec": label, "band": b, "odds_ratio": float(np.exp(res.params[k])),
                         "or_lo": float(ci.loc[k, 0]), "or_hi": float(ci.loc[k, 1]),
                         "p_value": float(res.pvalues[k]),
                         "gap_pp": float(np.mean(res.predict(rows_b) - res.predict(cf))),
                         "n_band": int(len(rows_b))})
        log.info("%-40s 80%%+ OR %.3f (%.0fs)", label, rows[-1]["odds_ratio"], time.time() - tm)
    reg = pd.DataFrame(rows)
    reg.to_csv(out / "neighbourhood_regression.csv", index=False)

    # ---- tract-level table for later HOLC linkage ---------------------------------
    tract = (df.groupby("census_tract")
               .agg(n=("approved", "size"), approval_rate=("approved", "mean"),
                    minority_pct=("tract_minority_population_percent", "first"),
                    tract_income_pct=("tract_to_msa_income_percentage", "first"),
                    state=("state_code", "first"), county=("county_code", "first")))
    tract.to_csv(out / "tract_table.csv")

    # ---- figures -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    within.plot(marker="o", ax=axes[0])
    axes[0].plot(range(len(raw)), raw["mean"], "k--", label="All applicants")
    axes[0].set_xticks(range(len(BAND_LABELS)), BAND_LABELS)
    axes[0].set_xlabel("Minority share of the census tract")
    axes[0].set_ylabel("Approval rate")
    axes[0].set_title("Approval rate by tract minority share")
    axes[0].legend(fontsize=8)
    for i, (label, s) in enumerate(reg.groupby("spec", sort=False)):
        axes[1].plot(s.band, 100 * s.gap_pp, marker="o", label=label)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xlabel("Tract minority share (compared with <10%)")
    axes[1].set_ylabel("Approval gap, percentage points")
    axes[1].set_title("Gap for similar applicants (logistic regression)")
    axes[1].legend(fontsize=8)
    savefig(fig, out / "fig_neighbourhood.png")

    top = reg[(reg.spec.str.startswith("+ fin")) & (reg.band == "80%+")].iloc[0].to_dict()
    save_json({"raw": raw.reset_index().to_dict("records"), "adjusted_80plus": top,
               "regression": reg.to_dict("records"), "n_tracts": int(len(tract)),
               "within_group": within.reset_index().to_dict("records")}, out / "summary.json")
    mark_done(NAME, t0)


if __name__ == "__main__":
    run()
