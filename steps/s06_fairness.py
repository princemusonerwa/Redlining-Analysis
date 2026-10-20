"""Step 06 - fairness of every model's predictions, by group, with 95% CIs.

Reference group is White (non-Hispanic). Metrics, all at a 0.5 threshold:
  SPD  statistical parity difference  = P(pred approve | group) - P(pred approve | White)
  DIR  disparate impact ratio         = P(pred approve | group) / P(pred approve | White)
  EOD  equal opportunity difference   = TPR(group) - TPR(White)   (TPR among truly approved)
  FPR_diff                             = FPR(group) - FPR(White)   (FPR among truly denied)
The model's gap is also compared with the ACTUAL gap in lender decisions, to
see whether the model narrows, copies, or widens it.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import time

import numpy as np
import pandas as pd

import config
from common import (GROUP_ORDER, get_logger, group_rates, load_prepared, mark_done,
                    pairwise_fairness, plt, save_json, savefig, step_dir)

NAME = "s06_fairness"
THR = 0.5


def fast_pair(y, p, g, focus, ref):
    """Vectorised SPD/DIR/EOD/FPR_diff for bootstrap resamples."""
    f, r = g == focus, g == ref
    def rates(m):
        sel = p[m].mean()
        tpr = p[m & (y == 1)].mean()
        fpr = p[m & (y == 0)].mean()
        return sel, tpr, fpr
    sf, tf, ff = rates(f)
    sr, tr, fr = rates(r)
    return {"SPD": sf - sr, "DIR": sf / sr, "EOD": tf - tr, "FPR_diff": ff - fr,
            "actual_gap": y[f].mean() - y[r].mean()}


def run():
    t0 = time.time()
    log = get_logger(NAME)
    out = step_dir(NAME)
    preds = pd.read_parquet(config.OUT / "s05_models" / "test_predictions.parquet")
    df = load_prepared()
    preds["sex"] = df["derived_sex"].astype(str).to_numpy()[preds["row"].to_numpy()]
    models = [c for c in preds.columns if c not in ("row", "approved", "group", "sex")]
    y = preds["approved"].to_numpy()
    g = preds["group"].astype(str).to_numpy()
    rng = np.random.default_rng(config.SEED)
    # Generate draws on demand to avoid retaining hundreds of full index arrays.

    rows, by_group = [], []
    for m in models:
        yhat = (preds[m].to_numpy() >= THR).astype(float)
        rates = group_rates(y, yhat, g)
        rates["model"] = m
        by_group.append(rates.reset_index().rename(columns={"index": "group"}))
        for focus in config.FOCUS_GROUPS:
            point = pairwise_fairness(rates, focus)
            boots = pd.DataFrame([fast_pair(y[i], yhat[i], g[i], focus, config.REFERENCE_GROUP) for i in (np.random.default_rng(config.SEED + b).integers(0, len(y), len(y)) for b in range(config.N_BOOT))])
            row = {"model": m, "group": focus, "n_group_test": int((g == focus).sum())}
            for k, v in point.items():
                row[k] = v
                if k in boots:
                    row[f"{k}_lo"], row[f"{k}_hi"] = boots[k].quantile([0.025, 0.975])
            rows.append(row)
        log.info("%-20s Black DIR %.3f", m, [r for r in rows if r["model"] == m and r["group"] == "Black"][0]["DIR"])

        # sex comparison: Female vs Male (single applicants only)
        sx = preds["sex"].to_numpy()
        sr = group_rates(y, yhat, sx)
        if {"Female", "Male"} <= set(sr.index):
            p = pairwise_fairness(sr, "Female", ref="Male")
            rows.append({"model": m, "group": "Female (vs Male)", "n_group_test": int((sx == "Female").sum()), **p})

    fair = pd.DataFrame(rows)
    fair.to_csv(out / "fairness_metrics.csv", index=False)
    by_group = pd.concat(by_group)
    by_group.to_csv(out / "rates_by_group.csv", index=False)

    from common import load_json
    best = load_json(config.OUT / "s05_models" / "summary.json")["best_model"]

    # ---- figures ----------------------------------------------------------
    bg = by_group[by_group.model == best].set_index("group")
    order = [x for x in GROUP_ORDER if x in bg.index]
    bg = bg.loc[order]
    fig, ax = plt.subplots(figsize=(9, 4))
    xs = np.arange(len(bg))
    ax.bar(xs - 0.2, bg.actual_approval_rate, 0.4, label="Actual lender decisions", color="#95a5a6")
    ax.bar(xs + 0.2, bg.predicted_approval_rate, 0.4, label=f"{best} predictions", color="#2c7fb8")
    ax.set_xticks(xs, order, rotation=30)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Approval rate")
    ax.set_title("Approval rates by group: real decisions vs the model")
    ax.legend()
    savefig(fig, out / "fig_actual_vs_predicted.png")

    fig, axes = plt.subplots(1, len(config.FOCUS_GROUPS), figsize=(13, 3.8), sharey=True)
    for ax, focus in zip(axes, config.FOCUS_GROUPS):
        s = fair[fair.group == focus].set_index("model")
        ax.errorbar(s.DIR, np.arange(len(s)), xerr=np.maximum(0, [s.DIR - s.DIR_lo, s.DIR_hi - s.DIR]), fmt="o", capsize=3)
        ax.axvline(0.8, color="#c0392b", ls="--", lw=1, label="0.80 descriptive reference")
        ax.axvline(s.actual_ratio.iloc[0], color="grey", ls=":", lw=1.5, label="actual decisions")
        ax.set_yticks(np.arange(len(s)), s.index)
        ax.set_title(f"{focus} vs White")
        ax.set_xlabel("Disparate impact ratio")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Disparate impact ratio per model (95% bootstrap CI)")
    savefig(fig, out / "fig_dir_by_model.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].bar(order, bg.loc[order, "tpr"], color="#27ae60")
    axes[0].set_ylim(0.8, 1.0)
    axes[0].set_title("True positive rate: approved applicants the model also approves")
    axes[1].bar(order, bg.loc[order, "fpr"], color="#e67e22")
    axes[1].set_title("False positive rate: denied applicants the model would approve")
    for a in axes:
        a.tick_params(axis="x", rotation=35)
    savefig(fig, out / "fig_error_rates.png")

    best_black = fair[(fair.model == best) & (fair.group == "Black")].iloc[0].to_dict()
    save_json({"best_model": best, "threshold": THR, "best_model_black_vs_white": best_black,
               "all": fair.to_dict("records")}, out / "summary.json")
    mark_done(NAME, t0)


if __name__ == "__main__":
    run()
