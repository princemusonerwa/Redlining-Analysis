"""Generate artificial data for integration tests; never research evidence."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
import numpy as np
import pandas as pd


def make_fixture(n=6000):
    rng = np.random.default_rng(753)
    cols = list(dict.fromkeys(['activity_year', 'lei', 'state_code', 'county_code', 'census_tract',
        'derived_race', 'derived_ethnicity', 'derived_sex', 'action_taken', 'tract_minority_population_percent']
        + config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES + config.LEAKY_FIELDS + ['derived_loan_product_type']))
    df = pd.DataFrame('1', index=range(n), columns=cols)
    profile = rng.integers(0, 6, n)
    for c in config.NUMERIC_FEATURES:
        df[c] = 100 + profile * 10
    df['income'] = 40 + profile * 20
    df['loan_amount'] = 150000 + profile * 40000
    df['property_value'] = 200000 + profile * 50000
    df['debt_to_income_ratio'] = np.take([15, 25, 33, 40, 45, 55], profile)
    df['loan_to_value_ratio'] = 65 + profile * 5
    df['loan_term'] = '360'
    df['activity_year'] = np.where(profile % 2, '2023', '2020')
    df['state_code'] = 'AL'
    df['county_code'] = np.where(profile % 2, '01001', '01003')
    df['census_tract'] = np.where(profile % 2, '01001020100', '01003020100')
    df['lei'] = rng.choice([f'LENDER{i:02}' for i in range(12)], n)
    df['derived_race'] = rng.choice(['White', 'Black or African American', 'Asian',
        'American Indian or Alaska Native', 'Native Hawaiian or Other Pacific Islander',
        '2 or more minority races', 'Joint', 'Race Not Available'], n)
    df['derived_ethnicity'] = rng.choice(['Hispanic or Latino', 'Not Hispanic or Latino', 'Ethnicity Not Available'], n, p=[.18,.8,.02])
    df['derived_sex'] = rng.choice(['Female', 'Male'], n)
    df['tract_minority_population_percent'] = rng.choice([5,20,40,65,90], n)
    df['tract_to_msa_income_percentage'] = rng.choice([60,80,100,120,140], n)
    df['applicant_age'] = '35-44'
    df['derived_dwelling_category'] = 'Single Family (1-4 Units):Site-Built'
    df['conforming_loan_limit'] = 'C'
    df['open-end_line_of_credit'] = '2'
    df['reverse_mortgage'] = '2'
    df['business_or_commercial_purpose'] = '2'
    probability = .78 - .02*profile - .04*(df.derived_race=='Black or African American')
    approved = rng.random(n) < probability
    action = np.where(approved, '1', '3')
    action[approved & (rng.random(n)<.08)] = '2'
    action[rng.random(n)<.04] = '4'
    df['action_taken'] = action
    df['interest_rate'] = np.where(action=='1', 3.0 + rng.random(n)*4, np.nan)
    df['rate_spread'] = np.where(action=='1', rng.normal(1.2,.8,n), np.nan)
    df['total_loan_costs'] = np.where(action=='1', rng.normal(4000,500,n), np.nan)
    df['origination_charges'] = np.where(action=='1', rng.normal(2000,300,n), np.nan)
    df['denial_reason-1'] = np.where(action=='3', '3', 'NA')
    df['derived_loan_product_type'] = 'Conventional:First Lien'
    df.loc[rng.choice(n, int(n*.01), replace=False), 'income'] = np.nan
    return df


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rows',type=int,default=6000)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    make_fixture(a.rows).to_csv(a.output,index=False)
    print(f'SYNTHETIC fixture written: {a.output}')
