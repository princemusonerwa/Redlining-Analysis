"""Shared helpers: step bookkeeping, I/O, logging, plotting defaults, metrics."""
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 140, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})

GROUP_ORDER = ["White", "Black", "Hispanic", "Asian", "Native American",
               "Pacific Islander", "Multiracial", "Joint", "Unknown"]


def step_dir(name):
    d = config.OUT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_logger(name):
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%H:%M:%S")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        log.addHandler(sh)
        fh = logging.FileHandler(step_dir(name) / "log.txt", mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log


def is_done(name):
    return (config.OUT / name / "_SUCCESS.json").exists()


def mark_done(name, started, extra=None):
    info = {"step": name, "finished": datetime.now().isoformat(timespec="seconds"),
            "seconds": round(time.time() - started, 1), "sample_n": config.SAMPLE_N}
    info.update(extra or {})
    save_json(info, config.OUT / name / "_SUCCESS.json")


def save_json(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        raise TypeError(type(o))
    Path(path).write_text(json.dumps(obj, indent=2, default=conv), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_prepared():
    return pd.read_parquet(config.OUT / "s04_prepare" / "prepared.parquet")


def savefig(fig, path):
    fig.savefig(path)
    plt.close(fig)


def stratified_subsample(df, n, by, seed=config.SEED):
    if len(df) <= n:
        return df
    frac = n / len(df)
    return (df.groupby(by, group_keys=False, observed=True)
              .apply(lambda g: g.sample(frac=frac, random_state=seed)))


# ---------------------------------------------------------------------------
# Fairness metrics
# ---------------------------------------------------------------------------
def group_rates(y_true, y_pred, groups):
    """Per-group selection rate, TPR, FPR, accuracy, plus the actual approval rate."""
    d = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_pred), "g": np.asarray(groups)})
    out = {}
    for g, s in d.groupby("g"):
        pos, neg = s[s.y == 1], s[s.y == 0]
        out[g] = {
            "n": len(s),
            "actual_approval_rate": s.y.mean(),
            "predicted_approval_rate": s.p.mean(),
            "tpr": pos.p.mean() if len(pos) else np.nan,
            "fpr": neg.p.mean() if len(neg) else np.nan,
            "accuracy": (s.y == s.p).mean(),
        }
    return pd.DataFrame(out).T


def pairwise_fairness(rates, focus, ref=config.REFERENCE_GROUP):
    r, f = rates.loc[ref], rates.loc[focus]
    return {
        "SPD": f.predicted_approval_rate - r.predicted_approval_rate,   # statistical parity difference
        "DIR": f.predicted_approval_rate / r.predicted_approval_rate,   # disparate impact ratio
        "EOD": f.tpr - r.tpr,                                           # equal opportunity difference
        "FPR_diff": f.fpr - r.fpr,
        "actual_gap": f.actual_approval_rate - r.actual_approval_rate,
        "actual_ratio": f.actual_approval_rate / r.actual_approval_rate,
    }


def bootstrap_ci(fn, n, n_boot=config.N_BOOT, seed=config.SEED):
    """Percentile CI for a statistic fn(idx) over row indices."""
    rng = np.random.default_rng(seed)
    vals = [fn(rng.integers(0, n, n)) for _ in range(n_boot)]
    vals = pd.DataFrame(vals)
    return vals.quantile(0.025), vals.quantile(0.975)


# ---------------------------------------------------------------------------
# Binned financial controls (missing values get their own bin), shared by the
# regression, matching, neighbourhood and pricing steps.
# ---------------------------------------------------------------------------
def _cut(s, edges, labels):
    b = pd.cut(s, edges, labels=labels, right=False).astype(str)
    return b.where(s.notna(), "missing")


def _q(s, k=10):
    b = pd.qcut(s, k, labels=False, duplicates="drop").astype("Int64").astype(str)
    return b.where(s.notna(), "missing")


FINANCE_CONTROLS = ["income_bin", "loan_amount_bin", "property_value_bin", "ltv_bin", "dti_bin",
                    "applicant_credit_score_type", "applicant_age"]
LOAN_CONTROLS = ["loan_type", "loan_purpose", "lien_status", "occupancy_type", "term_bin",
                 "total_units_bin", "construction_method", "open-end_line_of_credit",
                 "business_or_commercial_purpose", "preapproval", "submission_of_application",
                 "conforming_loan_limit", "activity_year"]
PLACE_CONTROLS = ["state_code", "tract_income_bin"]


def control_frame(df):
    c = pd.DataFrame(index=df.index)
    c["income_bin"] = _q(df["income"])
    c["loan_amount_bin"] = _q(df["loan_amount"])
    c["property_value_bin"] = _q(df["property_value"])
    c["ltv_bin"] = _cut(df["loan_to_value_ratio"], [-np.inf, 60, 80, 90, 95, 100.01, np.inf],
                        ["<60", "60-80", "80-90", "90-95", "95-100", ">100"])
    c["dti_bin"] = _cut(df["debt_to_income_ratio"], [-np.inf, 20, 30, 36, 43, 50, 60.01, np.inf],
                        ["<20", "20-30", "30-36", "36-43", "43-50", "50-60", ">60"])
    term = df["loan_term"]
    c["term_bin"] = np.where(term.isna(), "missing", np.where(term == 360, "360",
                             np.where(term == 180, "180", "other")))
    c["total_units_bin"] = np.where(df["total_units"].astype(str) == "1", "1", "2+")
    c["tract_income_bin"] = _q(df["tract_to_msa_income_percentage"], 5)
    for col in ["applicant_credit_score_type", "applicant_age", "state_code"] + \
            [x for x in LOAN_CONTROLS if x not in ("term_bin", "total_units_bin")]:
        c[col] = df[col].astype(str)
    return c


def dummies(ctrl, cols, min_count=50):
    """One-hot encode control columns (first level dropped), removing rare levels."""
    parts = []
    for col in cols:
        s = ctrl[col]
        vc = s.value_counts()
        s = s.where(s.isin(vc[vc >= min_count].index), "_rare")
        d = pd.get_dummies(s, prefix=col, drop_first=True, dtype="float32")
        parts.append(d)
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=ctrl.index)


def fmt_pct(x, d=1):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"


def make_group(race, eth):
    """Require explicit non-Hispanic ethnicity for race-specific groups."""
    g = race.map({"White": "White", "Black or African American": "Black", "Asian": "Asian",
        "American Indian or Alaska Native": "Native American",
        "Native Hawaiian or Other Pacific Islander": "Pacific Islander",
        "2 or more minority races": "Multiracial", "Joint": "Joint"}).fillna("Unknown")
    g = g.where(eth.eq("Not Hispanic or Latino").fillna(False), "Unknown")
    g = g.where(~eth.eq("Hispanic or Latino").fillna(False), "Hispanic")
    return pd.Categorical(g, categories=GROUP_ORDER)
