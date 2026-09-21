#!/usr/bin/env python3
"""
Turns a per-launch nsys export into the duration-histogram format used by
results/duration-histograms.json and blog/index.html's HIST_* charts.

Input: a CSV with one row per `_w8a8_triton_block_scaled_mm` kernel launch,
columns `duration_us,N,K,M`. Getting from a raw `.nsys-rep` capture to that
CSV is the nsys/DCGM-specific part (see this folder's README "How" section)
and isn't reproduced here -- this script picks up once you have per-launch
durations and shapes, and does the same binning/categorization this
project's analysis applied to produce every HIST_* chart in the blog.

    nsys export --type sqlite --tables='.*KERNEL.*|StringIds' capture.nsys-rep
    # then, against the resulting .sqlite (CUPTI_ACTIVITY_KIND_KERNEL joined
    # to StringIds for the kernel name, filtered to _w8a8_triton_block_scaled_mm,
    # with N/K/M recovered from the launch's grid dimensions and the known
    # SHAPES table below), write one row per launch to launches.csv, then:

    python3 build_histograms.py launches.csv --label c64_tuned > out.json
"""

import argparse
import csv
import json
import sys

# Same 69 log-spaced bin edges used throughout this project (24.2us to
# 2135.46us), so results here line up directly with results/duration-
# histograms.json and the blog's own charts.
BINS_US = [
    24.2, 25.8, 27.5, 29.32, 31.26, 33.33, 35.53, 37.88, 40.38, 43.05, 45.9,
    48.93, 52.16, 55.61, 59.29, 63.2, 67.38, 71.83, 76.58, 81.64, 87.04,
    92.79, 98.93, 105.46, 112.44, 119.87, 127.79, 136.23, 145.24, 154.84,
    165.07, 175.98, 187.61, 200.01, 213.23, 227.33, 242.35, 258.37, 275.45,
    293.65, 313.06, 333.75, 355.81, 379.33, 404.4, 431.13, 459.62, 490.0,
    522.39, 556.91, 593.72, 632.96, 674.8, 719.4, 766.95, 817.64, 871.68,
    929.29, 990.71, 1056.19, 1126.0, 1200.42, 1279.76, 1364.34, 1454.52,
    1550.65, 1653.14, 1762.4, 1878.89, 2003.07, 2135.46,
]

# Qwen/Qwen3.8-27B-FP8, tensor_parallel_size=2, NVIDIA L40S -- same 5 shapes
# as ../01-first-measurement/compare_default_vs_tuned.py.
SHAPES = {
    (17408, 5120): "gate_up_proj",
    (8192, 5120): "in_proj_qkvz",
    (7168, 5120): "qkv_proj",
    (5120, 8704): "down_proj",
    (5120, 3072): "out_proj",
}

# Coarse batch-size bands a launch's M falls into, matching the blog's
# BAND_LABELS ("M<=128" ... "M 2048-4096"). A launch with M > 4096 has no
# band (treated as unmapped) -- none of the real captures this project used
# ever saw one.
M_BAND_EDGES = [128, 256, 512, 1024, 2048, 4096]


def m_band(M):
    for i, edge in enumerate(M_BAND_EDGES):
        if M <= edge:
            return i
    return None


def bin_index(duration_us):
    if duration_us < BINS_US[0] or duration_us >= BINS_US[-1]:
        return None
    lo, hi = 0, len(BINS_US) - 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if BINS_US[mid] <= duration_us:
            lo = mid
        else:
            hi = mid - 1
    return lo


def build(rows):
    counts = {}  # (bin, shape, band) -> count
    for r in rows:
        duration_us = float(r["duration_us"])
        b = bin_index(duration_us)
        if b is None:
            continue
        shape = SHAPES.get((int(r["N"]), int(r["K"])), "unmapped")
        band = m_band(int(r["M"])) if shape != "unmapped" else None
        counts[(b, shape, band)] = counts.get((b, shape, band), 0) + 1

    durations = [float(r["duration_us"]) for r in rows]
    cats = [[b, shape, band, n] for (b, shape, band), n in sorted(counts.items())]
    return {
        "n": len(rows),
        "mean": round(sum(durations) / len(durations), 1) if durations else 0,
        "cats": cats,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_path")
    ap.add_argument("--label", required=True, help="e.g. c1_default, c64_tuned")
    args = ap.parse_args()

    with open(args.csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    result = {args.label: build(rows)}
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
