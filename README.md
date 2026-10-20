# HMDA mortgage lending disparity analysis

This project integrates data collection, conservative cleaning, exploratory analysis,
predictive models, fairness comparisons, adjusted approval analyses, neighborhood
and pricing analyses, SHAP explanations, and reports. It studies observed lender
decisions and associations. It does not establish causal discrimination.

## Install

Use Python 3.11 or 3.12 in a virtual environment. From this project directory:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux instead: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Use your existing raw data

Keep the original CSV unchanged. Point the runner to its actual location:

```bash
python run_all.py --raw "../hmda_random_sample_2020_2023.csv" --through s03
python run_all.py --raw "../hmda_random_sample_2020_2023.csv" --publish
```

The first command runs cleaning and EDA only. The second resumes and runs the
remaining analyses, then builds the tabbed report. No download occurs by default.
To use the default path instead, place the raw CSV at
`data/raw/hmda_random_sample_2020_2023.csv`.

## Download a new sample

```bash
python steps/s01_download.py --rows 1000000
# Increase or reduce concurrent worker tasks:
python steps/s01_download.py --rows 1000000 --workers 8
# Or download only if the default raw file is absent, then clean and explore:
python run_all.py --download --download-rows 1000000 --through s03
```

The downloader visits all 50 states, Washington, D.C., and Puerto Rico for each
year 2020-2023 by default. Pass `--states` to select a different coverage. It
allocates the target sample equally across state-year strata and samples rows
without replacement within each. It records source hashes, source counts, sample
counts, and weights in a manifest/CSV. The fixed seed reproduces samples for the
same source files. Changing source publications can change results.
Each worker downloads one source and samples it before moving on;
temporary sample shards are combined in state-year order. Four workers are used
by default; use `--workers` to adjust this for your network and disk capacity.
Strata smaller than their quota contribute all their rows; any shortfall is
reported, not silently filled by duplicating records.

This design differs from the original download notebook. **It is not a national
simple random sample.** Current models, EDA, fairness measures, regressions, and
confidence intervals remain unweighted. Weights are preserved for later survey
analysis but are not applied automatically. Do not present these outputs as
population estimates. Full state-year source files are downloaded to a temporary
directory and removed after a successful run; the final sample size does not
limit download size. Expect substantial disk, network, and runtime use. A failed
run removes its temporary source files, so rerunning downloads them again.
Existing final CSVs require explicit `--overwrite`.

## Steps and outputs

| Step | Responsibility | Main output |
|---|---|---|
| s01_download | Collect and sample raw records, explicit opt-in | data/raw/*.csv and manifest |
| s02_clean | Preserve rows, normalize values, flag invalid/duplicate records | outputs/s02_clean/cleaned.csv and summary.json |
| s03_eda | Coverage, missingness, distributions, approval comparisons | outputs/s03_eda/ tables and figures |
| s04_prepare | Define decided cohort, groups and categorical schema; field audit | outputs/s04_prepare/prepared.parquet |
| s05_models | Six models, training CV, held-out evaluation | metrics, predictions, saved models |
| s06_fairness | Prediction and error-rate disparities | fairness_metrics.csv |
| s07_same_finances | Regression ladder, lender effects, matching | regression_ladder.csv and matching.csv |
| s08_neighbourhood | Minority-share bands and tract summaries | neighbourhood_regression.csv and tract_table.csv |
| s09_pricing | Originated-loan price comparisons | pricing_regression.csv |
| s10_shap | Feature contributions and prediction gaps | figures, global_importance.csv, gap_decomposition.csv |
| s11_report | Assemble the saved findings with methodological context | REPORT.md, report.html, sections.json |

`publish_report.py` produces `outputs/published/hmda_fair_lending_report.html`.
Both HTML reports embed their figures. They are local reports, not deployed sites.
`config.py` owns paths/settings and `common.py` holds shared functions.
Original notebooks are retained unchanged under `notebooks/` as historical work;
they are not executed by the pipeline and retain their original local/Colab paths.

## Resume and selective runs

```bash
python run_all.py --raw "../hmda_random_sample_2020_2023.csv" --sample 50000 --through s05
python run_all.py --raw "../hmda_random_sample_2020_2023.csv" --from s07 --publish
python run_all.py --raw "../hmda_random_sample_2020_2023.csv" --only s10
python run_all.py --raw "../hmda_random_sample_2020_2023.csv" --force --publish
```

Supply the same `--raw`, `--sample` and output directory when resuming. Sample mode
subsamples raw records before cleaning and writes to `outputs_sample/`. The number
of decided applications can be smaller. `--from` reruns that step and later steps;
missing earlier prerequisites are computed first. `--only` requires current earlier
steps and invalidates later success markers. `--through` stops at a chosen step.

Success markers include a source-code/raw-content/settings fingerprint. Changed
input bytes, source code, or sample size invalidate cached work. Key output files
are checked for existence. These markers do not detect manual edits to output
files; use `--force` after editing outputs. Standalone step execution is for
advanced use and does not provide runner-level cache checks. Environment
variables `REDLINING_RAW` and `REDLINING_OUT` can set paths; CLI arguments override.

## Cleaning and modeling choices

- All records remain in the cleaned file. Approval calculations keep actions 1/2
  as approved, action 3 as denied. Actions 4-8 are outside that analysis cohort.
- Raw numeric text is retained in `raw_*` columns; `flag_*` columns identify
  exemptions, malformed values, fixed-rule invalid values and duplicate-like rows.
  Unknown categorical codes are not silently interpreted as observed numbers.
- Exact public-row matches are flagged but retained: public data cannot reliably
  distinguish duplicate ingestion from separate, identical-looking applications.
  Compare duplicate flags and sampling history before making a deletion decision.
- No global imputation, scaling or percentile capping occurs. Median imputation,
  missing indicators and scaling are fitted inside training folds for linear/KNN
  models. Signed logarithms retain zero and negative income.
- DTI ranges are mapped to representative values, including 15 for <20 and 65 for
  >60. These open-ended representatives are assumptions, not exact observations;
  raw values and approximation flags remain available for sensitivity analyses.
- Race-specific groups require explicit non-Hispanic ethnicity; Hispanic ethnicity
  takes precedence over race. Unknown ethnicity maps to Unknown. This is stricter
  than the original grouping and can change rates and counts.
- Predictive models exclude race, ethnicity, sex, tract minority share and selected
  decision-related fields. Age remains an input. Exclusion does not prove fairness
  or absence of proxy information. Integer codes for trees may impose order.
- The model used in fairness/reporting is chosen by training CV, not test score.
  SHAP selects a tree using CV and may prefer a similarly performing boosted tree
  for runtime. Its unit is probability for Random Forest and log-odds for boosting.
- Bootstrap intervals are not survey-design intervals. Approximate logistic
  marginal-gap intervals vary the group coefficient only. Matching coverage may
  be small; missing matches produce unavailable estimates.
- Historical document claims are not updated automatically. Mitigation,
  national reweighting, temporal/lender holdouts and HOLC linkage remain future work.

## Validation

Run focused checks and an end-to-end synthetic smoke run:

```bash
python -m unittest discover -s validation -p "test_*.py"
python validation/make_fixture.py --output validation/synthetic.csv
python run_all.py --raw validation/synthetic.csv --smoke --publish
```

Smoke mode reduces model sizes, folds, bootstrap draws and SHAP samples. It writes
to `outputs_smoke/`. **Its metrics are artificial validation output, not research
results.** Large full-data runtime and real-data statistical behavior still need
validation on your source CSV. Small/restricted samples may lack the groups,
minority-share bands, or outcome variation needed by adjusted analyses.

## Migrating from the previous project

Extract this package into a new directory first. Keep your raw CSV and previous
outputs as a record. Do not mix old output directories with this version.
The included `MIGRATION.md` maps old filenames to the new sequence. Notebook,
Word/PDF manuscript, petition and RFE contents have not been rewritten. Those
need separate review against the eventual full-data results. The screenshot's
existing `validation/` contents were not supplied and are not included here.