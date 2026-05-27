"""
Per-run analysis of 29-06-LF.

29-06-LF is a single continuous recording of four consecutive flow-loop runs
(run 1 clog-free; blockages confirmed by video in runs 2-4). Analysed as one
file the composite score is contaminated by a differential-pressure regime
change about 36 min in, so the file-level crossing is not a sound basis for a
lead time. This script splits the recording at the documented run timecodes
and analyses each run on its own, calibrating on that run's healthy opening.
It reproduces the per-run table in the Results chapter ("Per-Run Analysis of
29-06-LF").

The blockage for each run is taken as the differential-pressure spike that
marks the end of the run; the composite crossing uses the same 60 s rolling
median as the sigma sweep. Run 4 is reported for completeness but is excluded
from the thesis table because its blockage falls ~30 s before the recording
ends, leaving the smoothing window too little data to time the crossing.

Run: python3 split_2906_by_run.py
"""
import contextlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from app.batch import run_batch_analysis
from sigma_sweep import first_composite_crossing_min

SRC = Path(__file__).parent / "data" / "29-06-LF.xlsx"
SIGMA = 3.0
TAIL_SECONDS = 120  # extend each run to capture the blockage spike at its end

# Documented run timecodes (time-of-day). The logger's calendar date is wrong,
# but the time-of-day matches the lab log; the file spans 13:56:49-16:09:08.
RUNS = [
    ("Run 1 (no-block)", (14, 33, 5), (15, 14, 26)),
    ("Run 2 (blockage)", (15, 20, 20), (15, 28, 59)),
    ("Run 3 (blockage)", (15, 33, 24), (15, 44, 6)),
    ("Run 4 (blockage)", (15, 49, 3), (16, 8, 7)),
]

P_IN = "TS inlet pressure (Mean)"
P_OUT = "TS outlet pressure (Mean)"


def main() -> None:
    df = pd.read_excel(SRC)
    df["Time"] = pd.to_datetime(df["Time"])
    day = df["Time"].dt.normalize().iloc[0]

    header = f"{'Run':18}{'dur (min)':>10}{'blockage':>10}{'crossing':>10}{'lead':>9}"
    print(header)
    print("-" * len(header))
    for i, (name, start, end) in enumerate(RUNS, 1):
        a = day + pd.Timedelta(hours=start[0], minutes=start[1], seconds=start[2])
        b = (day + pd.Timedelta(hours=end[0], minutes=end[1], seconds=end[2])
             + pd.Timedelta(seconds=TAIL_SECONDS))
        seg = df[(df["Time"] >= a) & (df["Time"] <= b)].copy()
        if len(seg) < 150:
            print(f"{name:18} too short ({len(seg)} rows)")
            continue

        rel_min = ((seg["Time"] - seg["Time"].iloc[0]).dt.total_seconds() / 60.0).values
        dP = (seg[P_IN] - seg[P_OUT]).values
        blockage_min = float(rel_min[int(np.argmax(dP))])  # the end-of-run pressure spike

        tmp = f"/tmp/2906_run{i}.csv"
        seg.to_csv(tmp, index=False)
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            result = run_batch_analysis(tmp, sigma=SIGMA)
        crossing_min = first_composite_crossing_min(
            result["timeseries"], result["thresholds"]["fft_threshold"]
        )

        is_block = "blockage" in name
        if not is_block:
            blk_s, cross_s, lead_s = "--", str(crossing_min), ("no FP" if crossing_min is None else "FP")
        elif crossing_min is None:
            blk_s, cross_s, lead_s = f"{blockage_min:.2f}", "none", "miss"
        else:
            lead_s = f"{(blockage_min - crossing_min) * 60:+.0f} s"
            blk_s, cross_s = f"{blockage_min:.2f}", f"{crossing_min:.2f}"
        print(f"{name:18}{rel_min[-1]:10.1f}{blk_s:>10}{cross_s:>10}{lead_s:>9}")


if __name__ == "__main__":
    main()
