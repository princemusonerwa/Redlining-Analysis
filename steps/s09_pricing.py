"""Step 09 - pricing: among APPROVED, originated loans, do similar borrowers pay
different prices by group?

interest_rate and rate_spread cannot be used to predict approval (they leak),
but they are exactly the right outcomes for a pricing question. Sample:
originated (action 1), first-lien, closed-end, non-reverse loans with a
reported rate spread. Outcomes:
  rate_spread (percentage points over the market benchmark APOR)
  interest_rate
  high_cost   = rate_spread >= 1.5 points (descriptive indicator only; not a
                complete regulatory classification)
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
from s07_same_finances import GROUPS, group_dummies

NAME = "s09_pricing"


def run():
    t0 = time.time()
    log = get_logger(NAME)
    out = step_dir(NAME)
    df = load_prepared()
    m = ((df["action_taken"].astype(str) == "1") & (df["lien_status"].astype(str) == "1")
         & (df["open-end_line_of_credit"].astype(str) == "2") & (df["reverse_mortgage"].astype(str) == "2")
         & df["rate_spread"].notna() & df["interest_rate"].notna())
    d = df[m].reset_index(drop=True)
    if d.empty:
        raise ValueError("No originated first-lien closed-end non-reverse loans with reported pricing")
    d["high_cost"] = (d["rate_spread"] >= 1.5).astype(float)
    log.info("Pricing sample: %d originated first-lien loans", len(d))
    g = d["group"].astype(str)
    ctrl = control_frame(d)

    raw = d.groupby("group", observed=True)[["rate_spread", "interest_rate", "high_cost"]].mean()
    raw["n"] = d.groupby("group", observed=True).size()
    raw.to_csv(out / "raw_pricing_by_group.csv")

    G = group_dummies(g)
    full = dummies(ctrl, FINANCE_CONTROLS + LOAN_CONTROLS + PLACE_CONTROLS)
    rows = []
    for outcome in ["rate_spread", "interest_rate", "high_cost"]:
        for label, X in [("raw", G), ("+ finances, loan, year, state", pd.concat([G, full], axis=1))]:
            res = sm.OLS(d[outcome], sm.add_constant(X, has_constant="add")).fit(cov_type="HC1")
            ci = res.conf_int()
            for focus in GROUPS:
                k = f"grp_{focus}"
                rows.append({"outcome": outcome, "spec": label, "group": focus,
                             "diff_vs_white": float(res.params[k]), "lo": float(ci.loc[k, 0]),
                             "hi": float(ci.loc[k, 1]), "p_value": float(res.pvalues[k]),
                             "n": int(res.nobs), "r2": float(res.rsquared)})
        r = [x for x in rows if x["outcome"] == outcome and x["group"] == "Black"]
        log.info("%-14s Black - White: raw %.3f, adjusted %.3f", outcome, r[0]["diff_vs_white"], r[1]["diff_vs_white"])
    reg = pd.DataFrame(rows)
    reg.to_csv(out / "pricing_regression.csv", index=False)

    # Dollar translation: extra interest over the first 5 years of a typical loan
    med_loan = float(d["loan_amount"].median())
    adj = reg[(reg.outcome == "interest_rate") & (reg.spec != "raw")].set_index("group")["diff_vs_white"]
    dollars = {grp: float(med_loan * v / 100 * 5) for grp, v in adj.items()}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, outcome, title, scale, unit in [
        (axes[0], "rate_spread", "Rate spread vs White borrowers", 100, "basis points"),
        (axes[1], "high_cost", "Chance of spread >= 1.5 pp vs White", 100, "percentage points")]:
        s = reg[reg.outcome == outcome]
        piv = s.pivot(index="group", columns="spec", values="diff_vs_white").reindex(GROUPS) * scale
        lo = s.pivot(index="group", columns="spec", values="lo").reindex(GROUPS) * scale
        hi = s.pivot(index="group", columns="spec", values="hi").reindex(GROUPS) * scale
        xs = np.arange(len(GROUPS))
        for j, spec in enumerate(["raw", "+ finances, loan, year, state"]):
            ax.bar(xs + (j - 0.5) * 0.4, piv[spec], 0.4, yerr=[piv[spec] - lo[spec], hi[spec] - piv[spec]],
                   capsize=3, label=spec)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(xs, GROUPS, rotation=20)
        ax.set_ylabel(unit)
        ax.set_title(title)
        ax.legend(fontsize=8)
    savefig(fig, out / "fig_pricing.png")

    pick = lambda o, s: reg[(reg.outcome == o) & (reg.spec == s) & (reg.group == "Black")].iloc[0].to_dict()
    save_json({"n_loans": len(d), "raw_by_group": raw.reset_index().to_dict("records"),
               "black_rate_spread_raw": pick("rate_spread", "raw"),
               "black_rate_spread_adj": pick("rate_spread", "+ finances, loan, year, state"),
               "black_high_cost_adj": pick("high_cost", "+ finances, loan, year, state"),
               "median_loan_amount": med_loan, "extra_interest_5yr_dollars": dollars,
               "regression": reg.to_dict("records")}, out / "summary.json")
    mark_done(NAME, t0)


if __name__ == "__main__":
    run()
