"""Descriptive EDA of cleaned HMDA data. Requires pandas, numpy, matplotlib.

Rates are unweighted descriptions of the sample, not national estimates or
causal effects. Approved = action 1 or 2; denied = 3. All other actions are
excluded only from approval calculations. Missingness covers all retained rows.
No model is fitted. Existing EDA files are regenerated in the output directory.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def demographic_group(df):
    from common import make_group
    race = df.get('derived_race', pd.Series(pd.NA, index=df.index, dtype='string'))
    eth = df.get('derived_ethnicity', pd.Series(pd.NA, index=df.index, dtype='string'))
    return make_group(race, eth)


def rates(df, keys):
    t = df.groupby(keys, dropna=False, observed=True).approved.agg(['size', 'sum', 'mean'])
    return t.rename(columns={'size': 'n_decided', 'sum': 'n_approved', 'mean': 'approval_rate'}).reset_index()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=ROOT / 'outputs/s02_clean/cleaned.csv')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/s03_eda')
    a = p.parse_args(argv)
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.input, dtype=str, keep_default_na=False).replace('', pd.NA)
    base = [c for c in df if not c.startswith(('raw_', 'flag_'))]
    missing = df[base].isna().mean().sort_values(ascending=False).rename('missing_fraction')
    missing.to_csv(out / 'missingness.csv')
    df.action_taken.value_counts(dropna=False).rename('n').to_csv(out / 'action_counts.csv')
    df.groupby(['activity_year', 'state_code'], dropna=False).size().rename('n').to_csv(out / 'coverage.csv')
    d = df[df.action_taken.isin(['1', '2', '3'])].copy()
    d['approved'] = d.action_taken.isin(['1', '2']).astype(int)
    d['group'] = demographic_group(d)
    if d.empty:
        raise ValueError('No decided applications (action_taken 1, 2, 3)')
    comparisons = [('group', ['group']), ('year', ['activity_year']), ('state', ['state_code']),
                   ('year_group', ['activity_year', 'group'])]
    if 'derived_sex' in d:
        comparisons.append(('sex', ['derived_sex']))
    for name, keys in comparisons:
        table = rates(d, keys)
        table.to_csv(out / f'approval_by_{name}.csv', index=False)
        if len(keys) == 1:
            fig, ax = plt.subplots(figsize=(10, max(4, len(table) * .25)))
            ax.barh(table[keys[0]].astype('string').fillna('Missing'), table.approval_rate * 100)
            ax.set(xlabel='Sample approval rate (%)', xlim=(0, 100), title=f'Approval by {name} (unweighted)')
            fig.tight_layout()
            fig.savefig(out / f'approval_by_{name}.png', dpi=150)
            plt.close(fig)
    numeric = [c for c in ['income', 'loan_amount', 'property_value', 'debt_to_income_ratio',
                          'loan_to_value_ratio', 'interest_rate', 'tract_minority_population_percent'] if c in df]
    if numeric:
        numbers = df[numeric].apply(pd.to_numeric, errors='coerce')
        numbers.describe(percentiles=[.01, .25, .5, .75, .99]).to_csv(out / 'numeric_summary.csv')
        for c in numeric:
            values = numbers[c].dropna()
            if values.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(values, bins=50)
            ax.set(title=f'{c}: all nonmissing values', xlabel=c, ylabel='Sample records')
            if c == 'income':
                ax.set_xlabel('Income (thousands of dollars)')
            fig.tight_layout()
            fig.savefig(out / f'distribution_{c}.png', dpi=150)
            plt.close(fig)
    if 'income' in d:
        income = pd.to_numeric(d.income, errors='coerce')
        if income.nunique() > 1:
            d['income_band'] = pd.qcut(income, 4, duplicates='drop').astype('string').fillna('Missing')
            rates(d, ['income_band', 'group']).to_csv(out / 'approval_by_income_group.csv', index=False)
    summary = {'all_rows': len(df), 'decided_rows': len(d), 'excluded_from_approval_rates': len(df)-len(d),
               'approval_rate': float(d.approved.mean()),
               'interpretation': 'Unweighted sample descriptions; not causal or national estimates. Duplicates retained.'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'EDA saved to {out}; {len(d):,} decided applications.')


def run():
    import time
    import config
    from common import mark_done
    started = time.time()
    main(["--input", str(config.CLEAN_CSV), "--output-dir", str(config.OUT / "s03_eda")])
    mark_done("s03_eda", started)


if __name__ == '__main__':
    main()
