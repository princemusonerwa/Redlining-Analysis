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
         's07_same_finances', 's08_neighbourhood', 's09_pricing', 's10_shap',
         's11_geospatial', 's12_report']
# Files used to detect missing artifacts before a resume.
ARTIFACTS = {'s02_clean': ['cleaned.csv', 'summary.json'], 's03_eda': ['summary.json', 'approval_by_group.csv'],
    's04_prepare': ['prepared.parquet', 'summary.json'],
    's05_models': ['test_metrics.csv', 'test_predictions.parquet', 'summary.json', 'test_idx.npy'],
    's06_fairness': ['fairness_metrics.csv', 'summary.json'],
    's07_same_finances': ['regression_ladder.csv', 'matching.csv', 'summary.json'],
    's08_neighbourhood': ['neighbourhood_regression.csv', 'summary.json'],
    's09_pricing': ['pricing_regression.csv', 'summary.json'],
    's10_shap': ['global_importance.csv', 'summary.json'],
    's11_geospatial': [], 's12_report': ['REPORT.md', 'report.html']}


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
    p.add_argument('--publish', action='store_true', help='Build the tabbed HTML report after s12_report')
    p.add_argument('--geospatial', action='store_true',
                   help='Run optional s11_geospatial stage before the report')
    a = p.parse_args()
    if a.sample < 0:
        p.error('--sample must be nonnegative')
    geospatial_boundaries = ROOT / 'data/geo/cb_2023_us_tract_500k.shp'
    geospatial_year = 2023
    if a.geospatial and not geospatial_boundaries.is_file():
        p.error(f'Census tract boundary shapefile not found: {geospatial_boundaries}')
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
    geo_step = 's11_geospatial'
    if (only == geo_step or start == geo_step or end == geo_step) and not a.geospatial:
        p.error('Selecting s11_geospatial requires the --geospatial option.')
    if start and STEPS.index(start) > STEPS.index(end):
        p.error('--from must not be after --through')
    selected = [only] if only else STEPS[:STEPS.index(end)+1]
    config.OUT.mkdir(parents=True, exist_ok=True)
    run_key = signature(config.RAW_CSV) + f':{config.SAMPLE_N}:{config.SMOKE}'

    def stage_run_key(step):
        if step != geo_step:
            return run_key
        boundary_files = sorted(geospatial_boundaries.parent.glob(
            geospatial_boundaries.stem + '.*'))
        boundary_state = [
            (str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
            for path in boundary_files if path.is_file()
        ]
        options = json.dumps({
            'year': geospatial_year,
            'boundaries': boundary_state,
        }, sort_keys=True)
        return f'{run_key}:geospatial:{hashlib.sha256(options.encode()).hexdigest()}'

    def artifacts_for(step):
        if step == geo_step:
            suffix = geospatial_year
            return [
                f'tract_approval_map_{suffix}.html',
                f'tract_summary_{suffix}.csv',
                f'unmatched_tracts_{suffix}.csv',
                f'map_audit_{suffix}.json',
            ]
        return ARTIFACTS[step]

    def ready(s):
        marker = config.OUT / s / '_SUCCESS.json'
        try:
            return json.loads(marker.read_text()).get('run_key') == stage_run_key(s) and all(
                (config.OUT / s / f).exists() for f in artifacts_for(s))
        except (OSError, ValueError):
            return False
    if only:
        prerequisites = (
            ['s04_prepare'] if only == geo_step
            else [s for s in STEPS[:STEPS.index(only)] if s != geo_step]
        )
        missing = [s for s in prerequisites if not ready(s)]
        if missing:
            p.error(f'Prerequisites missing/stale: {", ".join(missing)}. Run the required earlier steps first.')
    for s in selected:
        if s == geo_step and not a.geospatial:
            print(f'== {s}: not requested; skipping', flush=True)
            continue
        forced = a.force or only or (start and STEPS.index(s) >= STEPS.index(start))
        if not forced and ready(s):
            print(f'== {s}: current; skipping', flush=True)
            continue
        # An upstream rerun invalidates all downstream success markers.
        for later in STEPS[STEPS.index(s):]:
            (config.OUT / later / '_SUCCESS.json').unlink(missing_ok=True)
        print(f'== {s}: running', flush=True)
        begin = time.time()
        if s == geo_step:
            if not ready('s04_prepare'):
                p.error('Geospatial mapping requires a current s04_prepare output. Run through s04 first.')
            geospatial_out = config.OUT / geo_step
            command = [
                sys.executable,
                str(ROOT / 'steps/s11_geospatial.py'),
                '--input', str(config.OUT / 's04_prepare/prepared.parquet'),
                '--boundaries', str(geospatial_boundaries),
                '--output', str(geospatial_out),
            ]
            subprocess.run(command, check=True)
            missing = [
                str(config.OUT / geo_step / filename)
                for filename in artifacts_for(s)
                if not (config.OUT / geo_step / filename).is_file()
            ]
            if missing:
                p.error(f'Geospatial step did not create expected outputs: {", ".join(missing)}')
        else:
            importlib.import_module(s).run()
        save_json({'step': s, 'run_key': stage_run_key(s), 'seconds': round(time.time()-begin, 2),
                   'sample_n': config.SAMPLE_N, 'smoke': config.SMOKE}, config.OUT / s / '_SUCCESS.json')
    if a.publish:
        required_steps = [s for s in STEPS if s != geo_step]
        if not all(ready(s) for s in required_steps):
            p.error('Publishing requires a complete current run through s12_report.')
        subprocess.run([sys.executable, str(ROOT / 'publish_report.py')], check=True)
    print(f'Outputs: {config.OUT}', flush=True)


if __name__ == '__main__':
    main()
