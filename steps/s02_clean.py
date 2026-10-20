"""Conservative HMDA cleaning; preserves rows and writes an audit.

No imputation, scaling, percentile capping, or outcome filtering occurs here.
Keep missingness visible. Data-derived transformations belong in training folds.
Exact public-row matches are flagged, NOT dropped: distinct applications can
have identical anonymized public fields. Identifier-based duplicates also remain
for review. Columns named raw_* preserve changed numeric text.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


ROOT = Path(__file__).resolve().parents[1]
NUMERIC = ['loan_amount', 'income', 'property_value', 'loan_to_value_ratio',
           'debt_to_income_ratio', 'loan_term', 'interest_rate', 'rate_spread',
           'total_loan_costs', 'origination_charges', 'tract_minority_population_percent',
           'tract_to_msa_income_percentage', 'ffiec_msa_md_median_family_income',
           'tract_population', 'tract_owner_occupied_units', 'tract_one_to_four_family_homes',
           'tract_median_age_of_housing_units', 'sample_weight']
MISSING = {'', 'na', 'n/a', 'nan', 'null', 'none'}
DTI = {'<20%': 15.0, '20%-<30%': 25.0,
       '30%-<36%': 33.0, '50%-60%': 55.0, '>60%': 65.0}


def clean(df):
    df = df.copy()
    required = {'action_taken', 'activity_year', 'state_code'}
    if required - set(df.columns):
        raise ValueError(
            f'Missing columns: {sorted(required-set(df.columns))}')
    original_columns = list(df.columns)
    for col in original_columns:
        s = df[col].astype('string').str.strip()
        df[col] = s.mask(s.str.lower().isin(MISSING))
    public = [c for c in original_columns if c not in [
        'source_record_id', 'sample_weight']]
    df['flag_exact_public_duplicate'] = df.duplicated(public, keep=False)
    if 'source_record_id' in df:
        df['flag_source_id_duplicate'] = df.source_record_id.notna(
        ) & df.source_record_id.duplicated(keep=False)
    for col in NUMERIC:
        if col not in df:
            continue
        raw = df[col].copy()
        df['raw_' + col] = raw
        # Numeric field exemptions are preserved separately, never interpreted as zero.
        exempt = raw.str.lower().eq('exempt').fillna(False)
        df['flag_' + col + '_exempt'] = exempt
        s = raw.mask(exempt)
        values = pd.to_numeric(s, errors='coerce').astype(float)
        if col == 'debt_to_income_ratio':
            mapped = s.map(DTI)
            df['flag_dti_approximated'] = mapped.notna()
            values = values.fillna(mapped)
        invalid = (s.notna() & values.isna()) | (
            values.notna() & ~np.isfinite(values))
        if col in ['loan_amount', 'property_value', 'loan_term', 'sample_weight']:
            invalid |= values <= 0
        if col in ['loan_to_value_ratio', 'debt_to_income_ratio', 'interest_rate',
                   'tract_population', 'tract_owner_occupied_units', 'tract_one_to_four_family_homes',
                   'tract_median_age_of_housing_units']:
            invalid |= values < 0
        if col == 'tract_minority_population_percent':
            invalid |= (values < 0) | (values > 100)
        # Negative income and rate spread can be meaningful: retain them.
        df['flag_' + col + '_invalid'] = invalid.fillna(False)
        df[col] = values.mask(invalid)
    # Normalize numeric-looking codes while preserving strings and leading zeros.
    for col, width in [('county_code', 5), ('census_tract', 11)]:
        if col in df:
            s = df[col].str.replace(r'\.0$', '', regex=True)
            valid = s.str.fullmatch(r'\d{1,' + str(width) + r'}').fillna(False)
            df['raw_' + col] = df[col]
            df['flag_' + col + '_invalid'] = s.notna() & ~valid
            df[col] = s.where(valid).str.zfill(width)
    for col in ['activity_year', 'action_taken']:
        df[col] = df[col].str.replace(r'\.0$', '', regex=True)
    df['flag_action_taken_invalid'] = df.action_taken.notna(
    ) & ~df.action_taken.isin(list('12345678'))
    df['state_code'] = df.state_code.str.upper()
    flags = [c for c in df if c.startswith('flag_')]
    audit = {'rows_input': len(df), 'rows_output': len(df), 'rows_removed': 0,
             'flag_counts': {c: int(df[c].sum()) for c in flags},
             'note': 'Duplicates retained; DTI range representatives are approximations, not observed exact ratios.'}
    return df, audit


def run():
    import time
    import config
    from common import step_dir, mark_done, save_json

    started = time.time()

    if not config.RAW_CSV.exists():
        raise FileNotFoundError(
            f"Raw CSV not found: {config.RAW_CSV}. "
            "Check config.RAW_CSV or run the download step first."
        )

    df = pd.read_csv(
        config.RAW_CSV,
        dtype=str,
        keep_default_na=False,
    )
    n_raw = len(df)

    if config.SAMPLE_N and len(df) > config.SAMPLE_N:
        df = df.sample(
            n=config.SAMPLE_N,
            random_state=config.SEED,
        ).reset_index(drop=True)

    df, audit = clean(df)

    audit.update(
        raw_file_rows=n_raw,
        input_file=str(config.RAW_CSV),
        sample_n=config.SAMPLE_N,
    )

    out = step_dir("s02_clean")
    out.mkdir(parents=True, exist_ok=True)
    config.CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(config.CLEAN_CSV, index=False)
    save_json(audit, out / "summary.json")

    mark_done("s02_clean", started)

    print(f"Saved {len(df):,} cleaned rows to {config.CLEAN_CSV}")
    print(f"Saved cleaning summary to {out / 'summary.json'}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=ROOT /
                   'data/raw/hmda_random_sample_2020_2023.csv')
    p.add_argument('--output', type=Path, default=ROOT /
                   'outputs/s02_clean/cleaned.csv')
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    if a.output.resolve() == a.input.resolve():
        p.error('Input and output must differ')
    if a.output.exists() and not a.overwrite:
        p.error('Output exists; use --overwrite explicitly')
    df, audit = clean(pd.read_csv(a.input, dtype=str, keep_default_na=False))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    temp = a.output.with_suffix('.csv.part')
    df.to_csv(temp, index=False)
    temp.replace(a.output)
    a.output.with_suffix('.audit.json').write_text(json.dumps(audit, indent=2))
    (a.output.parent / 'summary.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(
        f'Saved {len(df):,} rows to {a.output}; no rows automatically removed.')


if __name__ == '__main__':
    main()
