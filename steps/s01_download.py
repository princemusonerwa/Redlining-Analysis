"""Download a reproducible, stratified HMDA sample optimized for ultra-large datasets.

Uses native urllib streams within a ThreadPoolExecutor to prevent fsspec connection 
mismatch errors on Windows environments. Includes network pacing and error logs.
"""
import os
import time
import argparse
import threading
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from tqdm import tqdm

# Constants
STATES = 'AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY'.split()
YEARS = ['2020', '2021', '2022', '2023']

# Thread-safe lock for writing to the single shared CSV file
file_write_lock = threading.Lock()


def fetch_and_sample_stratum(year, state, quota, seed, output_file, row_pbar):
    """Downloads via native urllib streams, downsamples chunks, and appends to disk."""
    url = (
        "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
        f"?years={year}&states={state}"
    )

    # Generate a unique deterministic seed sequence for this specific stratum
    state_index = STATES.index(state) if state in STATES else 99
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, int(year), state_index]))

    headers = {'User-Agent': 'HMDA-research/1.0', 'Accept': 'text/csv'}
    req = urllib.request.Request(url, headers=headers)

    # Built-in transient network retry layer
    for attempt in range(3):
        try:
            # Open a native connection stream over the wire
            with urllib.request.urlopen(req, timeout=120) as response:
                # Feed the direct response stream into pandas chunk by chunk
                chunks = pd.read_csv(
                    response,
                    dtype=str,
                    keep_default_na=False,
                    chunksize=100_000,
                    low_memory=False
                )

                retained = None
                count = 0

                for chunk in chunks:
                    if not chunk['activity_year'].eq(str(year)).all() or not chunk['state_code'].eq(state).all():
                        return None, f"Malformed data integrity issue in {state}-{year}"

                    # Track absolute index markers safely
                    chunk['source_record_id'] = [
                        f'{year}-{state}-{i}' for i in range(count, count + len(chunk))]
                    count += len(chunk)

                    # Assign uniform random float weight priority numbers
                    chunk['_priority'] = rng.random(len(chunk))

                    # Accumulate and aggressively filter down to your target quota allocation on the fly
                    retained = pd.concat([retained, chunk], ignore_index=True).nsmallest(
                        quota, '_priority')

                if retained is None or retained.empty:
                    return None, f"Empty source received for {state}-{year}"

                # Drop tracking priorities and calculate sampling design weight values
                retained = retained.drop(columns='_priority')
                retained['sample_weight'] = count / len(retained)

                # Safely block other threads and stream chunk straight to physical hard drive
                with file_write_lock:
                    write_header = not os.path.exists(output_file)
                    retained.to_csv(output_file, mode='a',
                                    index=False, header=write_header)

                # Update the global progress bar thread-safely
                row_pbar.update(len(retained))
                return len(retained), None

        except Exception as e:
            if attempt == 2:
                return None, f"Failure for {state}-{year} after 3 attempts: {str(e)}"
            # Stagger retries with an exponential backoff sleep
            time.sleep(3 ** attempt)


def main(target_rows, output_file, max_workers, seed):
    strata = [(y, s) for y in YEARS for s in STATES]
    quota_per_stratum = target_rows // len(strata)

    # Clean up old partial remnants to avoid corrupted appends
    if os.path.exists(output_file):
        print(f"Removing pre-existing file: {output_file}")
        os.remove(output_file)

    os.makedirs(os.path.dirname(os.path.abspath(output_file))
                if os.path.dirname(output_file) else '.', exist_ok=True)

    print(
        f"\nTargeting: {target_rows:,} total rows ({quota_per_stratum:,} rows allocation per state-year combo)")
    print(
        f"Opening {max_workers} concurrent pipeline workers...\n", flush=True)

    # Initialize the Global Master Progress Bars
    with tqdm(total=target_rows, unit=' rows', desc='[1/2] Total Rows Saved', colour='green') as row_pbar:
        with tqdm(total=len(strata), unit=' files', desc='[2/2] Files Processed ', colour='cyan') as file_pbar:

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}

                for year, state in strata:
                    # Subtle millisecond pacing trick during initialization to prevent network adapter shock
                    time.sleep(0.05)
                    job = executor.submit(
                        fetch_and_sample_stratum, year, state, quota_per_stratum, seed, output_file, row_pbar)
                    futures[job] = (year, state)

                for future in as_completed(futures):
                    res = future.result()
                    if res is not None:
                        rows_saved, error = res
                        if error:
                            file_pbar.write(f"[Warning] {error}")
                    file_pbar.update(1)

    if os.path.exists(output_file):
        final_size_bytes = os.path.getsize(output_file)
        print(f"\n{'='*50}\nExecution Complete Successfully!")
        print(f"Output Saved To: {output_file}")
        print(
            f"Total File Size on Disk: {final_size_bytes / (1024**3):.2f} GB")
        print(f"{'='*50}")
    else:
        print("\nAll pipeline worker processing threads failed to salvage a valid dataset frame.")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    # Adjust to 100000000 for your full scale
    p.add_argument('--rows', type=int, default=1000000)
    p.add_argument('--output', type=str,
                   default=str(Path(__file__).resolve().parents[1] / 'data' / 'raw' /
                               'hmda_random_sample_2020_2023.csv'))
    p.add_argument('--workers', type=int, default=5,
                   help='Lowered to 5 to protect router/DNS capacity')
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()

    start_time = time.time()
    main(a.rows, a.output, a.workers, a.seed)

    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(
        f"Total Pipeline Runtime Execution Duration: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
