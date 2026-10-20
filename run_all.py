"""Run the integrated pipeline. Downloading is explicit; existing raw data is reused."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'steps')]
STEPS = ['s02_clean', 's03_eda', 's04_prepare', 's05_models', 's06_fairness',
         's07_same_finances', 's08_neighbourhood', 's09_pricing', 's10_shap', 's11_report']
# Files used to detect missing artifacts before a resume.
ARTIFACTS = {'s02_clean': ['cleaned.csv', 'summary.json'], 's03_eda': ['summary.json', 'approval_by_group.csv'],
    's04_prepare': ['prepared.parquet', 'summary.json'],
    's05_models': ['test_metrics.csv', 'test_predictions.parquet', 'summary.json', 'test_idx.npy'],
    's06_fairness': ['fairness_metrics.csv', 'summary.json'],
    's07_same_finances': ['regression_ladder.csv', 'matching.csv', 'summary.json'],
    's08_neighbourhood': ['neighbourhood_regression.csv', 'summary.json'],
    's09_pricing': ['pricing_regression.csv', 'summary.json'],
    's10_shap': ['global_importance.csv', 'summary.json'], 's11_report': ['REPORT.md', 'report.html']}


def resolve(prefix, parser):
    matches = [s for s in STEPS if s.startswith(prefix)]
    if len(matches) != 1:
        parser.error(f'Unknown or ambiguous step {prefix!r}. Choose one of: {", ".join(STEPS)}')
    return matches[0]


def signature(raw):
    h = hashlib.sha256()
    for path in [ROOT / 'config.py', ROOT / 'common.py', ROOT / 'run_all.py', ROOT / 'publish_report.py', ROOT / 'requirements.txt', *sorted((ROOT / 'steps').glob('*.py'))]:
        h.update(path.name.encode()); h.update(path.read_bytes())
    # Content hash makes resume safe even when input sizes/timestamps coincide.
    with raw.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    h.update(str(raw.resolve()).encode())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--from', dest='start')
    mode.add_argument('--only')
    p.add_argument('--through', help='Stop after this step (e.g. s03 for cleaning and EDA)')
    p.add_argument('--raw', type=Path, help='Existing raw CSV, including your original sample')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--sample', type=int, default=0)
    p.add_argument('--smoke', action='store_true', help='Reduced model settings for validation ONLY')
    p.add_argument('--force', action='store_true')
    p.add_argument('--download', action='store_true', help='Download if the raw CSV does not exist')
    p.add_argument('--download-rows', type=int, default=1000000)
    p.add_argument('--publish', action='store_true', help='Build the tabbed HTML report after s11')
    a = p.parse_args()
    if a.sample < 0:
        p.error('--sample must be nonnegative')
    os.environ['REDLINING_SAMPLE'] = str(a.sample)
    os.environ['REDLINING_SMOKE'] = '1' if a.smoke else '0'
    if a.raw:
        os.environ['REDLINING_RAW'] = str(a.raw.resolve())
    if a.output_dir:
        os.environ['REDLINING_OUT'] = str(a.output_dir.resolve())
    elif a.smoke:
        os.environ['REDLINING_OUT'] = str(ROOT / 'outputs_smoke')
    import config
    from common import save_json
    if a.download and not config.RAW_CSV.exists():
        subprocess.run([sys.executable, str(ROOT / 'steps/s01_download.py'), '--output', str(config.RAW_CSV),
                        '--rows', str(a.download_rows)], check=True)
    if not config.RAW_CSV.exists():
        p.error(f'Raw CSV not found: {config.RAW_CSV}. Supply --raw PATH or --download.')
    start = resolve(a.start, p) if a.start else None
    only = resolve(a.only, p) if a.only else None
    end = resolve(a.through, p) if a.through else STEPS[-1]
    if only and a.through:
        p.error('--only cannot be combined with --through')
    if start and STEPS.index(start) > STEPS.index(end):
        p.error('--from must not be after --through')
    selected = [only] if only else STEPS[:STEPS.index(end)+1]
    config.OUT.mkdir(parents=True, exist_ok=True)
    run_key = signature(config.RAW_CSV) + f':{config.SAMPLE_N}:{config.SMOKE}'
    def ready(s):
        marker = config.OUT / s / '_SUCCESS.json'
        try:
            return json.loads(marker.read_text()).get('run_key') == run_key and all(
                (config.OUT / s / f).exists() for f in ARTIFACTS[s])
        except (OSError, ValueError):
            return False
    if only:
        missing = [s for s in STEPS[:STEPS.index(only)] if not ready(s)]
        if missing:
            p.error(f'Prerequisites missing/stale: {", ".join(missing)}. Run through {STEPS[STEPS.index(only)-1]} first.')
    for s in selected:
        forced = a.force or only or (start and STEPS.index(s) >= STEPS.index(start))
        if not forced and ready(s):
            print(f'== {s}: current; skipping', flush=True)
            continue
        # An upstream rerun invalidates all downstream success markers.
        for later in STEPS[STEPS.index(s):]:
            (config.OUT / later / '_SUCCESS.json').unlink(missing_ok=True)
        print(f'== {s}: running', flush=True)
        begin = time.time()
        importlib.import_module(s).run()
        save_json({'step': s, 'run_key': run_key, 'seconds': round(time.time()-begin, 2),
                   'sample_n': config.SAMPLE_N, 'smoke': config.SMOKE}, config.OUT / s / '_SUCCESS.json')
    if a.publish:
        if not all(ready(s) for s in STEPS):
            p.error('Publishing requires a complete current run through s11_report.')
        subprocess.run([sys.executable, str(ROOT / 'publish_report.py')], check=True)
    print(f'Outputs: {config.OUT}', flush=True)


if __name__ == '__main__':
    main()
