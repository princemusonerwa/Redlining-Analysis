"""Step 11 - assemble every step's saved results into one report.

Writes outputs/s11_report/REPORT.md (figures linked from the step folders) and
outputs/s11_report/report.html (self-contained: figures embedded, ready to publish).
Interpretation sentences are generated from the numbers, so they stay correct
when the pipeline is re-run.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import base64
import html
import time
from datetime import date

import numpy as np
import pandas as pd

import config
from common import save_json
from common import load_json, mark_done, step_dir

NAME = "s11_report"


def pp(x, d=1, sign=True):
    return f"{100 * x:+.{d}f}" if sign else f"{100 * x:.{d}f}"


def less_more(x, d=1):
    """-0.083 -> '8.3 percentage points less likely'"""
    return f"{abs(100 * x):.{d}f} percentage points {'less' if x < 0 else 'more'} likely"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


class Report:
    """Collects blocks once, renders to Markdown and HTML."""

    def __init__(self):
        self.blocks = []

    def h(self, text, anchor):
        self.blocks.append(("h", text, anchor))

    def p(self, text):
        self.blocks.append(("p", text))

    def note(self, text):
        self.blocks.append(("note", text))

    def bullets(self, items):
        self.blocks.append(("ul", items))

    def table(self, df, caption=None):
        self.blocks.append(("table", df, caption))

    def fig(self, path, caption):
        self.blocks.append(("fig", path, caption))

    # -- markdown ------------------------------------------------------------
    def to_md(self, title, subtitle, out_dir):
        lines = [f"# {title}", "", f"*{subtitle}*", ""]
        for b in self.blocks:
            kind = b[0]
            if kind == "h":
                lines += [f"## {b[1]}", ""]
            elif kind == "p":
                lines += [b[1], ""]
            elif kind == "note":
                lines += [f"> {b[1]}", ""]
            elif kind == "ul":
                lines += [f"- {x}" for x in b[1]] + [""]
            elif kind == "table":
                if b[2]:
                    lines += [f"**{b[2]}**", ""]
                lines += [md_table(b[1]), ""]
            elif kind == "fig":
                rel = b[1].relative_to(out_dir.parent).as_posix()
                lines += [f"![{b[2]}](../{rel})", "", f"*{b[2]}*", ""]
        return "\n".join(lines)

    # -- html --------------------------------------------------------------------
    def to_html(self, title, subtitle, meta):
        toc = "".join(f'<li><a href="#{b[2]}">{html.escape(b[1])}</a></li>'
                      for b in self.blocks if b[0] == "h")
        body, fig_n, open_section = [], 0, False
        for b in self.blocks:
            kind = b[0]
            if kind == "h":
                if open_section:
                    body.append("</section>")
                body.append(f'<section id="{b[2]}"><h2>{html.escape(b[1])}</h2>')
                open_section = True
            elif kind == "p":
                body.append(f"<p>{inline(b[1])}</p>")
            elif kind == "note":
                body.append(f'<p class="note">{inline(b[1])}</p>')
            elif kind == "ul":
                body.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in b[1]) + "</ul>")
            elif kind == "table":
                cap = f"<caption>{html.escape(b[2])}</caption>" if b[2] else ""
                body.append(f'<div class="tbl">{b[1].to_html(index=False, border=0, escape=True).replace("<table", "<table" + "", 1).replace("<thead>", cap + "<thead>", 1)}</div>')
            elif kind == "fig":
                fig_n += 1
                data = base64.b64encode(b[1].read_bytes()).decode()
                body.append(f'<figure><div class="plate"><img alt="{html.escape(b[2])}" '
                            f'src="data:image/png;base64,{data}"></div>'
                            f"<figcaption><span>Figure {fig_n}</span> {html.escape(b[2])}</figcaption></figure>")
        if open_section:
            body.append("</section>")
        return HTML_TEMPLATE.format(title=html.escape(title), subtitle=html.escape(subtitle),
                                    meta=meta, toc=toc, body="\n".join(body))


def md_table(df):
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def inline(text):
    """Escape, then allow **bold** markup."""
    t = html.escape(text)
    parts = t.split("**")
    return "".join(f"<strong>{p}</strong>" if i % 2 else p for i, p in enumerate(parts))


HTML_TEMPLATE = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>
<style>
/* Layout: a single reading column with a sticky contents rail on wide screens; figures sit on light plates. */
:root {{
  --bg: #f5f6f8; --surface: #ffffff; --ink: #1d2433; --muted: #5b6475; --rule: #d9dde5;
  --accent: #a8382c;   /* HOLC grade D "hazardous" red, used sparingly */
  --accent-soft: #f3e3e0; --plate: #ffffff;
  --display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --body: "Source Serif 4", Georgia, "Times New Roman", serif;
  --mono: "IBM Plex Mono", ui-monospace, Consolas, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #12161f; --surface: #1a202c; --ink: #e5e8ef; --muted: #9aa3b5; --rule: #2d3544;
  --accent: #e0796c; --accent-soft: #3a2422; --plate: #f4f5f7; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #12161f; --surface: #1a202c; --ink: #e5e8ef; --muted: #9aa3b5; --rule: #2d3544;
  --accent: #e0796c; --accent-soft: #3a2422; --plate: #f4f5f7; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--ink); font: 17px/1.6 var(--body); }}
.wrap {{ max-width: 1120px; margin: 0 auto; padding-block: 40px 80px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: grid; grid-template-columns: 220px minmax(0, 1fr); gap: 48px; }}
header {{ grid-column: 1 / -1; border-bottom: 2px solid var(--ink); padding-bottom: 20px; }}
.eyebrow {{ font: 600 12px/1 var(--display); letter-spacing: .12em; text-transform: uppercase; color: var(--accent); }}
h1 {{ font: 700 clamp(30px, 4.6vw, 46px)/1.08 var(--display); margin: 10px 0 8px; text-wrap: balance; letter-spacing: -.01em; }}
.sub {{ color: var(--muted); font-style: italic; margin: 0 0 14px; max-width: 65ch; }}
.meta {{ font: 13px/1.5 var(--mono); color: var(--muted); display: flex; flex-wrap: wrap; gap: 6px 22px; }}
nav {{ position: sticky; top: calc(env(safe-area-inset-top, 0px) + 20px); align-self: start; font: 500 14px/1.4 var(--display); }}
nav ol {{ list-style: none; padding: 0; margin: 0; display: grid; gap: 8px; counter-reset: s; }}
nav li {{ counter-increment: s; }}
nav a {{ color: var(--muted); text-decoration: none; display: block; padding-left: 10px; border-left: 2px solid var(--rule); }}
nav a:hover, nav a:focus-visible {{ color: var(--ink); border-left-color: var(--accent); outline: none; }}
main {{ min-width: 0; }}
section {{ margin-bottom: 56px; }}
h2 {{ font: 600 26px/1.2 var(--display); margin: 0 0 14px; text-wrap: balance; padding-top: 10px; border-top: 1px solid var(--rule); }}
p, ul {{ max-width: 68ch; }}
li {{ margin-bottom: 6px; }}
strong {{ font-weight: 600; }}
.note {{ background: var(--accent-soft); border-left: 3px solid var(--accent); padding: 12px 16px; font-size: 15.5px; }}
figure {{ margin: 26px 0; }}
.plate {{ background: var(--plate); border: 1px solid var(--rule); border-radius: 4px; padding: 12px; }}
.plate img {{ display: block; margin: 0 auto; max-width: 100%; height: auto; }}
figcaption {{ font: 14px/1.45 var(--body); color: var(--muted); margin-top: 8px; max-width: 75ch; }}
figcaption span {{ font: 600 12px var(--display); letter-spacing: .08em; text-transform: uppercase; color: var(--ink); margin-right: 6px; }}
.tbl {{ overflow-x: auto; margin: 18px 0 24px; }}
table {{ border-collapse: collapse; font: 13.5px/1.4 var(--mono); font-variant-numeric: tabular-nums; min-width: 100%; }}
caption {{ text-align: left; font: 600 13px var(--display); letter-spacing: .06em; text-transform: uppercase; color: var(--muted); padding-bottom: 8px; }}
th {{ text-align: left; font: 600 13px var(--display); border-bottom: 2px solid var(--ink); padding: 6px 12px 6px 0; white-space: nowrap; }}
td {{ border-bottom: 1px solid var(--rule); padding: 6px 12px 6px 0; white-space: nowrap; }}
@media (max-width: 860px) {{ .wrap {{ grid-template-columns: minmax(0, 1fr); gap: 28px; }} nav {{ position: static; }} }}
@media (prefers-reduced-motion: no-preference) {{ html {{ scroll-behavior: smooth; }} }}
</style>
<div class="wrap">
<header><div class="eyebrow">Fair-lending analysis · HMDA 2020–2023</div><h1>{title}</h1><p class="sub">{subtitle}</p><div class="meta">{meta}</div></header>
<nav aria-label="Contents"><ol>{toc}</ol></nav>
<main>
{body}
</main>
</div>
"""


def run():
    """Render saved outputs, with no hard-coded directional conclusions."""
    t0 = time.time()
    out = step_dir(NAME)
    O = config.OUT
    s1 = load_json(O / "s04_prepare/summary.json")
    s2 = load_json(O / "s05_models/summary.json")
    s7 = load_json(O / "s10_shap/summary.json")
    clean = load_json(O / "s02_clean/summary.json")
    R = Report()
    sections = []
    def section(title, key, explanation, tables=(), figures=()):
        R.h(title, key)
        R.p(explanation)
        entry = {"title": title, "key": key, "explanation": explanation, "tables": [], "figures": []}
        for relative, caption in tables:
            table = pd.read_csv(O / relative)
            R.table(table.round(4), caption)
            entry["tables"].append({"path": relative, "caption": caption})
        for relative, caption in figures:
            if (O / relative).exists():
                R.fig(O / relative, caption)
                entry["figures"].append({"path": relative, "caption": caption})
        sections.append(entry)

    label = "SYNTHETIC / SMOKE VALIDATION ONLY. " if config.SMOKE else ""
    if config.SAMPLE_N:
        label += f"Quick run using at most {config.SAMPLE_N:,} raw records. "
    section("Overview", "overview", label +
        f"This run uses {s1['n_raw']:,} cleaned sample records and {s1['n_analysis']:,} decided applications. "
        f"The sample approval rate is {s1['approval_rate']:.1%}. "
        f"The model selected by training cross-validation is {s2['best_model']}. "
        "All rates, models and regressions are unweighted sample analyses. Sampling weights, if present, "
        "are not used downstream. These results do not establish national rates or causal discrimination.")
    section("Collection and cleaning", "cleaning",
        f"Input: {config.RAW_CSV.name}. Cleaning inspected {clean['rows_input']:,} records and retained "
        f"{clean['rows_output']:,}. Fixed validity rules flag malformed numeric entries and preserve raw numeric text. "
        "Duplicate public records are flagged and retained because public fields do not uniquely identify applications. "
        "No whole-dataset imputation, scaling or percentile capping is performed. DTI range representatives are "
        "approximations. Existing source datasets may use different sampling designs; inspect the source manifest.")
    section("Exploratory analysis", "eda",
        "Missingness and distributions describe all cleaned records. Approval comparisons use only action codes "
        "1 and 2 (approved) and 3 (denied). Other actions are excluded from those denominators. "
        "The White reference requires explicitly non-Hispanic ethnicity; unknown ethnicity is not treated as White.",
        [("s03_eda/coverage.csv", "Sample coverage by year and state"),
         ("s03_eda/missingness.csv", "Missingness before modeling"),
         ("s03_eda/approval_by_group.csv", "Unadjusted approval rates")],
        [("s03_eda/approval_by_group.png", "Sample approval rates by group"),
         ("s03_eda/approval_by_year.png", "Sample approval rates by year")])
    section("Preparation and leakage diagnostics", "preparation",
        "Approval models exclude selected decision-related fields and direct protected-group fields. "
        "The single-feature AUC audit uses the same observations to construct and score bins: it is exploratory, "
        "not held-out performance. The separate LightGBM comparison uses held-out data. "
        "Absence of the listed fields does not establish that all possible leakage or proxies are absent.",
        [("s04_prepare/leakage_audit.csv", "Descriptive field audit")],
        [("s04_prepare/fig_leakage_audit.png", "In-sample single-field diagnostics"),
         ("s05_models/fig_leakage_demo.png", "Held-out comparison with and without rate fields")])
    section("Predictive modeling", "models",
        f"Training uses {s2['n_train']:,} observations; the held-out test set has {s2['n_test']:,}. "
        f"Cross-validation uses {config.N_FOLDS} folds. Imputation and scaling are fitted within training folds. "
        "Signed log transforms retain negative income. Trees use categorical integer codes, which can impose "
        "an artificial order. The selected model maximizes mean training CV ROC-AUC; test scores are reported "
        "for comparison. Targets are recorded lender decisions, not independently verified creditworthiness.",
        [("s05_models/cv_summary.csv", "Training cross-validation"),
         ("s05_models/test_metrics.csv", "Held-out metrics for all six models")],
        [("s05_models/fig_cv_metrics.png", "Cross-validation metrics"),
         ("s05_models/fig_roc.png", "Test ROC curves")])
    section("Model fairness", "fairness",
        "SPD compares predicted approval rates by subtraction; DIR compares them by division. EOD compares "
        "true-positive rates and FPR_diff compares false-positive rates relative to recorded decisions. "
        "The prediction threshold is 0.5. Bootstrap intervals measure sample variability under row resampling. "
        "The 0.80 line is a descriptive reference, not a legal determination. Sex comparisons have point estimates only.",
        [("s06_fairness/fairness_metrics.csv", "Group fairness metrics and available intervals")],
        [("s06_fairness/fig_actual_vs_predicted.png", "Recorded outcomes and predicted outcomes"),
         ("s06_fairness/fig_dir_by_model.png", "Ratios across models"),
         ("s06_fairness/fig_error_rates.png", "Model error rates")])
    section("Adjusted approval comparisons", "adjusted",
        "The regression ladder adds financial, loan/year, place, and lender controls. Matching compares "
        "applicants in shared bins; check the matched share before interpreting a result. Gaps stored as "
        "gap_pp are probability differences: multiply by 100 to obtain percentage points. "
        "Credit-score type is a scoring-system identifier, not an actual credit score. Remaining differences "
        "are associations conditional on observed controls. Logistic gap intervals vary the group coefficient "
        "only and are approximate; lender-model intervals use clustered covariance.",
        [("s07_same_finances/regression_ladder.csv", "Regression ladder"),
         ("s07_same_finances/matching.csv", "Matching and overlap")],
        [("s07_same_finances/fig_gap_ladder.png", "Adjusted and matched comparisons")])
    section("Neighborhood comparisons", "neighborhood",
        "Comparisons use tract minority-share bands, with under 10% as the reference. Adjustments add "
        "own group, observed finances, loan/year, state and tract income. Tract aggregation pools years; "
        "tract boundaries and demographic definitions should be harmonized before historical map linkage. "
        "HOLC linkage is not implemented.",
        [("s08_neighbourhood/neighbourhood_regression.csv", "Neighborhood regressions")],
        [("s08_neighbourhood/fig_neighbourhood.png", "Observed and adjusted neighborhood differences")])
    section("Loan pricing", "pricing",
        "The pricing sample contains originated first-lien, closed-end, non-reverse loans with reported "
        "interest rates and spreads. OLS differences use heteroskedasticity-robust standard errors. "
        "The internal high_cost variable means spread at least 1.5 percentage points; it is a descriptive "
        "indicator, not a complete regulatory classification. Pricing is conditional on origination and "
        "reported pricing, so selection can affect comparisons. Dollar translations ignore amortization.",
        [("s09_pricing/pricing_regression.csv", "Pricing differences relative to White borrowers")],
        [("s09_pricing/fig_pricing.png", "Unadjusted and adjusted pricing")])
    section("Model explanations", "explanations",
        f"SHAP explains {s7['model']} on the {s7['output_unit']} scale. "
        "The explained tree is selected using training CV; a similarly performing boosted model may be "
        "preferred for runtime. Contributions explain model predictions, not causes of lender behavior.",
        [("s10_shap/global_importance.csv", "Mean absolute feature contributions"),
         ("s10_shap/gap_decomposition.csv", "Mean contribution differences between groups")],
        [("s10_shap/fig_shap_importance.png", "Global importance"),
         ("s10_shap/fig_shap_beeswarm.png", "Distribution of feature contributions"),
         ("s10_shap/fig_shap_gap_decomposition.png", "Group prediction-gap contributions"),
         ("s10_shap/fig_shap_dependence.png", "Feature dependence"),
         ("s10_shap/fig_shap_waterfall_denied.png", "An individual prediction")])
    section("Limitations and unfinished work", "limitations",
        "Public HMDA omits actual credit scores and full credit histories. Sampling may be unequal and "
        "uncertainty estimates do not account for the full survey design. Retained duplicate-like records, "
        "missingness and unobserved variables require sensitivity analyses. A random row split is not "
        "an out-of-year or out-of-lender validation. Bias mitigation, calibration-by-group evaluation, "
        "proxy tests, survey-weighted estimates and historical HOLC linkage remain unimplemented. "
        "Older manuscript results must be reconciled against a fresh full-data run.")
    title = "Mortgage Lending Disparity Analysis"
    subtitle = label + "Integrated collection, cleaning, EDA, modeling and disparity analysis"
    (out / "REPORT.md").write_text(R.to_md(title, subtitle, out), encoding="utf-8")
    (out / "report.html").write_text(R.to_html(title, subtitle, ""), encoding="utf-8")
    save_json({"title": title, "subtitle": subtitle, "sections": sections}, out / "sections.json")
    mark_done(NAME, t0)


if __name__ == "__main__":
    run()
