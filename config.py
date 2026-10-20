"""Central settings for the analysis pipeline. Every step imports from here."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_CSV = Path(os.environ.get("REDLINING_RAW", ROOT / "data/raw/hmda_random_sample_2020_2023.csv")).resolve()

# Quick-test mode: set REDLINING_SAMPLE=50000 to run everything on a subsample,
# written to a separate outputs folder so it never mixes with the full run.
SAMPLE_N = int(os.environ.get("REDLINING_SAMPLE", "0") or 0)
OUT = Path(os.environ.get("REDLINING_OUT", ROOT / ("outputs_sample" if SAMPLE_N else "outputs"))).resolve()
CLEAN_CSV = OUT / "s02_clean" / "cleaned.csv"
SMOKE = os.environ.get("REDLINING_SMOKE") == "1"
PIPELINE_VERSION = "3.0"

SEED = 42
TEST_SIZE = 0.20
N_FOLDS = 5
N_BOOT = 300            # bootstrap repetitions for confidence intervals
CV_MAX_ROWS = 200_000   # cross-validation runs on a stratified subsample of train this size
KNN_MAX_ROWS = 100_000  # KNN is O(n^2); trained on a subsample this size
SHAP_ROWS = 8_000

# ---------------------------------------------------------------------------
# Outcome definition (HMDA action_taken codes)
#   1 originated, 2 approved-not-accepted  -> approved (1)
#   3 denied                                -> denied (0)
#   4 withdrawn, 5 incomplete, 6 purchased, 7/8 preapproval -> excluded
# ---------------------------------------------------------------------------
APPROVED_CODES = {"1", "2"}
DENIED_CODES = {"3"}

# Group used for fairness comparisons. Hispanic of any race is its own group,
# which is the convention in the fair-lending literature.
REFERENCE_GROUP = "White"
FOCUS_GROUPS = ["Black", "Hispanic", "Asian"]

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "loan_amount", "income", "property_value", "loan_to_value_ratio",
    "debt_to_income_ratio", "loan_term",
    "tract_to_msa_income_percentage", "ffiec_msa_md_median_family_income",
    "tract_owner_occupied_units", "tract_one_to_four_family_homes",
    "tract_median_age_of_housing_units", "tract_population",
]
CATEGORICAL_FEATURES = [
    "loan_type", "loan_purpose", "lien_status", "occupancy_type",
    "construction_method", "total_units", "derived_dwelling_category",
    "conforming_loan_limit", "applicant_credit_score_type",
    "submission_of_application", "preapproval", "business_or_commercial_purpose",
    "open-end_line_of_credit", "reverse_mortgage", "applicant_age",
    "activity_year",
]

# Never given to the approval model: they are only recorded AFTER a decision
# (or are the decision itself), so they leak the answer.
LEAKY_FIELDS = [
    "interest_rate", "rate_spread", "total_loan_costs", "origination_charges",
    "purchaser_type", "hoepa_status", "initially_payable_to_institution",
    "denial_reason-1",
]

# Protected / demographic fields: kept for fairness analysis, never model inputs.
# tract_minority_population_percent is a direct demographic proxy, so it is
# also excluded from the model and studied separately in step 05.
PROTECTED = ["group", "derived_race", "derived_ethnicity", "derived_sex",
             "tract_minority_population_percent"]

# Smoke mode exercises integration only; never use its metrics as research results.
if SMOKE:
    N_FOLDS = 2
    N_BOOT = 20
    CV_MAX_ROWS = 2000
    KNN_MAX_ROWS = 2000
    SHAP_ROWS = 120
