#!/usr/bin/env python3
"""
Turn a flat (dur_us, cat, band) list (as saved by build_cache_aware_mband.py /
build_groupfix_mband.py) into a SHAPE_MBAND_DATA-compatible series object
(n, median, mean, bin_cats), using the exact same log-spaced bin edges as
the original chart (05's, carried through unmodified in 20) -- so the new
series is directly comparable/stackable with the existing ones.
"""
import json

BINS_PATH = "/tmp/shape_mband_bins.json"
DURS_PATH = "/tmp/cache_aware_c64_durs.json"
OUT_PATH = "/tmp/cache_aware_series.json"


def bin_index(bins, v):
    # same convention as the chart's log-scale binning: clamp into [0, len(bins)-2]
    lo, hi = bins[0], bins[-1]
    if v <= lo:
        return 0
    if v >= hi:
        return len(bins) - 2
    # linear scan is fine at this scale (70 bins)
    for i in range(len(bins) - 1):
        if bins[i] <= v < bins[i + 1]:
            return i
    return len(bins) - 2


def main():
    bins = json.load(open(BINS_PATH))
    durs = json.load(open(DURS_PATH))

    n = len(durs)
    all_d = sorted(d for d, _, _ in durs)
    mean = sum(all_d) / n
    median = all_d[n // 2]

    counts = {}
    for d, cat, band in durs:
        bi = bin_index(bins, d)
        key = (bi, cat, band)
        counts[key] = counts.get(key, 0) + 1

    bin_cats = [[bi, cat, band, cnt] for (bi, cat, band), cnt in counts.items()]

    series = {"n": n, "median": round(median, 1), "mean": round(mean, 1), "bin_cats": bin_cats}
    with open(OUT_PATH, "w") as f:
        json.dump(series, f)
    print(f"n={n} mean={mean:.2f} median={median:.2f} bin_cats={len(bin_cats)}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
