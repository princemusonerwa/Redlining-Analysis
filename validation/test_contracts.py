"""Focused tests for sampling, cleaning, preparation boundaries and report inputs."""
import importlib
import io
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'steps')]
from s01_download import (
    DEFAULT_STATES, combine_sample_shards, download_and_sample, fetch, publish_outputs, sample_csv,
)
from s02_clean import clean
from common import make_group


class Contracts(unittest.TestCase):
    def test_default_download_coverage_includes_all_states_dc_and_puerto_rico(self):
        self.assertEqual(len(DEFAULT_STATES), 52)
        self.assertEqual(len(set(DEFAULT_STATES)), 52)
        self.assertIn('DC', DEFAULT_STATES)
        self.assertIn('PR', DEFAULT_STATES)

    def test_workers_fetch_sample_and_combine_in_stratum_order(self):
        with tempfile.TemporaryDirectory() as folder:
            barrier = threading.Barrier(2)
            strata = [(2020, 'AL'), (2020, 'AK')]
            work_dir = Path(folder) / 'work'
            work_dir.mkdir()

            def write_source(_url, path):
                barrier.wait(timeout=5)
                state = path.stem.split('_')[1]
                pd.DataFrame({
                    'activity_year': ['2020', '2020'],
                    'state_code': [state, state],
                    'action_taken': ['1', '3'],
                    'value': [1, 2],
                }).to_csv(path, index=False)

            with patch('s01_download.fetch', side_effect=write_source):
                results = download_and_sample(
                    strata, rows=2, seed=42, work_dir=work_dir, workers=2
                )

            output = Path(folder) / 'combined.csv'
            combine_sample_shards([result['path'] for result in results], output)
            combined = pd.read_csv(output, dtype=str)
            self.assertEqual(combined['state_code'].tolist(), ['AL', 'AK'])

    def test_publish_outputs_keeps_existing_csv_if_manifest_cannot_be_staged(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'sample.csv'
            sample_temp = Path(folder) / 'sample.csv.part'
            output.write_text('previous sample')
            sample_temp.write_text('new sample')

            with self.assertRaises(TypeError):
                publish_outputs(sample_temp, output, {'invalid': object()})

            self.assertEqual(output.read_text(), 'previous sample')
            self.assertTrue(sample_temp.exists())
            self.assertFalse(Path(folder, 'sample.manifest.json').exists())

    def test_fetch_downloads_source_each_run(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'source.csv'
            payload = b'activity_year,state_code,action_taken\n2020,AL,1\n'

            with patch('s01_download.urlopen', return_value=io.BytesIO(payload)) as open_url:
                fetch('https://example.test/source.csv', path)
                open_url.assert_called_once()

            self.assertEqual(
                set(pd.read_csv(path, nrows=0).columns),
                {'activity_year', 'state_code', 'action_taken'},
            )
            updated_payload = b'activity_year,state_code,action_taken\n2021,AK,1\n'
            with patch('s01_download.urlopen', return_value=io.BytesIO(updated_payload)) as open_url:
                fetch('https://example.test/source.csv', path)
                open_url.assert_called_once()
            self.assertIn('2021,AK,1', path.read_text())

    def test_fetch_does_not_retry_forbidden_response(self):
        from urllib.error import HTTPError

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'source.csv'
            forbidden = HTTPError('https://example.test/source.csv', 403, 'Forbidden', {}, None)
            with patch('s01_download.urlopen', side_effect=forbidden) as open_url:
                with self.assertRaises(HTTPError):
                    fetch('https://example.test/source.csv', path)
            open_url.assert_called_once()

    def test_seeded_without_replacement_and_chunk_independence(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'source.csv'
            pd.DataFrame({'activity_year': ['2020'] * 40, 'state_code': ['AL'] * 40,
                          'value': range(40)}).to_csv(p, index=False)
            a, n = sample_csv(p, 11, 42, 2020, 'AL', chunksize=7)
            b, _ = sample_csv(p, 11, 42, 2020, 'AL', chunksize=13)
            self.assertEqual(n, 40)
            self.assertEqual(set(a.source_record_id), set(b.source_record_id))
            self.assertTrue(a.source_record_id.is_unique)
            self.assertAlmostEqual(a.sample_weight.iloc[0], 40 / 11)
            all_rows, _ = sample_csv(p, 100, 42, 2020, 'AL')
            self.assertEqual(len(all_rows), 40)

    def test_cleaning_retains_rows_and_flags_without_imputing(self):
        raw = pd.DataFrame({'activity_year': ['2020']*5, 'state_code': ['AL']*5,
            'action_taken': ['1','2','3','4','4'], 'loan_amount': ['100','-5','bad','100','100'],
            'income': ['-10','NA','50','100','100'], 'debt_to_income_ratio': ['<20%','Exempt','bad','40','40'],
            'census_tract': ['1001020100']*5})
        d, audit = clean(raw)
        self.assertEqual(len(d), len(raw))
        self.assertEqual(audit['rows_removed'], 0)
        self.assertEqual(d.income.iloc[0], -10)
        self.assertTrue(pd.isna(d.income.iloc[1]))
        self.assertTrue(d.flag_loan_amount_invalid.iloc[1])
        self.assertEqual(d.debt_to_income_ratio.iloc[0], 15)
        self.assertTrue(d.flag_debt_to_income_ratio_exempt.iloc[1])
        self.assertTrue(d.flag_debt_to_income_ratio_invalid.iloc[2])
        self.assertEqual(d.census_tract.iloc[0], '01001020100')
        self.assertTrue(d.flag_exact_public_duplicate.iloc[3])
        self.assertTrue(d.flag_exact_public_duplicate.iloc[4])

    def test_reference_group_requires_known_ethnicity(self):
        g = make_group(pd.Series(['White']*4), pd.Series([
            'Not Hispanic or Latino','Hispanic or Latino','Ethnicity Not Available',None]))
        self.assertEqual(list(g), ['White','Hispanic','Unknown','Unknown'])

    def test_signed_transform_handles_negative_income(self):
        from s05_models import signed_log1p
        z = signed_log1p(np.array([-100,0,100]))
        self.assertTrue(np.isfinite(z).all())
        self.assertAlmostEqual(z[0], -z[2])

    def test_all_stages_have_runner_entrypoint(self):
        from run_all import STEPS
        for name in STEPS:
            self.assertTrue(callable(importlib.import_module(name).run), name)


if __name__ == '__main__':
    unittest.main()
