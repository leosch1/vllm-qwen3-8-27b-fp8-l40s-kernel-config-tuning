#!/usr/bin/env python3
"""
Build a cache-aware-tuned (18 round 2 / 17's v2 output) c=64 SHAPE_MBAND_DATA
series, directly comparable to experiment 05's chart and to 20's groupfix
panel.

Unlike 20's groupfix, this config is NOT a single-parameter patch on top of
`tuned` -- it's a real, independently-generated 5-shape config from the
patched autotune script (different BLOCK_SIZE_M/N, num_warps, num_stages at
many M anchors, not just GROUP_SIZE_M). So its real launches do NOT have
byte-identical grid/block signatures to the original tuned capture, and
20/build_groupfix_mband.py's shortcut (reuse tuned's classification) does
NOT apply here. This script rebuilds the classification from the
cache-aware config's own JSONs instead -- same underlying method
(18-out-down-proj-order-disambiguation/resolve_order.py's real
program-order disambiguation for the out_proj/down_proj collision, which is
about vLLM's execution order and is independent of which config is
deployed).

Input: a CSV already extracted on-cluster (start,end,gridX,blockX,deviceId,
globalPid) from CUPTI_ACTIVITY_KIND_KERNEL for the
_w8a8_triton_block_scaled_mm kernel, via extract_kernels.py (same procedure,
new capture).
"""
import csv
import json
import math
from pathlib import Path

CACHE_AWARE_DIR = Path(__file__).resolve().parents[1] / "17-cache-aware-autotune-rerun/cache-aware-tuned-configs-v2"
CSV_PATH = "/tmp/cache_aware_c64_launches.csv"

SHAPES = {
    "gate_up_proj": (17408, 5120),
    "in_proj_qkvz": (8192, 5120),
    "qkv_proj": (7168, 5120),
    "out_proj": (5120, 3072),
    "down_proj": (5120, 8704),
}

BAND_LABELS = ["M ≤ 128", "M 128–256", "M 256–512", "M 512–1024", "M 1024–2048", "M 2048–4096"]


def band_for_m(m):
    if m <= 128:
        return 0
    if m <= 256:
        return 1
    if m <= 512:
        return 2
    if m <= 1024:
        return 3
    if m <= 2048:
        return 4
    if m <= 4096:
        return 5
    return None


def _load_config(n, k):
    fn = f"{CACHE_AWARE_DIR}/N={n},K={k},device_name=NVIDIA_L40S,dtype=fp8_w8a8,block_shape=[128,128].json"
    return {int(m): v for m, v in json.load(open(fn)).items()}


def build_shape_labels(max_m=4096):
    cfgs = {name: _load_config(n, k) for name, (n, k) in SHAPES.items()}

    def nearest_anchor(anchors, m):
        return min(anchors, key=lambda a: abs(a - m))

    def gridx_blockx(name, m):
        n, k = SHAPES[name]
        cfg_by_anchor = cfgs[name]
        anchors = sorted(cfg_by_anchor)
        cfg = cfg_by_anchor[nearest_anchor(anchors, m)]
        gx = math.ceil(m / cfg["BLOCK_SIZE_M"]) * math.ceil(n / cfg["BLOCK_SIZE_N"])
        return gx, cfg["num_warps"] * 32

    pair = frozenset({"out_proj", "down_proj"})

    def label_for(shape_set):
        if len(shape_set) == 1:
            return next(iter(shape_set))
        return "out_proj_or_down_proj" if frozenset(shape_set) == pair else "unmapped"

    raw = {}
    key_shape_ms = {}
    for name, (n, _k) in SHAPES.items():
        for m in range(1, max_m + 1):
            key = gridx_blockx(name, m)
            raw.setdefault(key, set()).add(name)
            key_shape_ms.setdefault(key, {}).setdefault(name, []).append(m)

    labels = {key: label_for(s) for key, s in raw.items()}
    return labels, key_shape_ms


def fetch_launches(csv_path, labels):
    launches = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            gx, bx = int(row["gridX"]), int(row["blockX"])
            cat = labels.get((gx, bx), "unmapped")
            launches.append({
                "start": int(row["start"]),
                "end": int(row["end"]),
                "dev": row["deviceId"],
                "gpid": row["globalPid"],
                "gx": gx, "bx": bx,
                "cat": cat,
            })
    return launches


def resolve_by_order(seq):
    n = len(seq)
    resolved = [None] * n
    for i, launch in enumerate(seq):
        if launch["cat"] != "out_proj_or_down_proj":
            continue
        prev_is_gate = i > 0 and seq[i - 1]["cat"] == "gate_up_proj"
        next_is_gate = i < n - 1 and seq[i + 1]["cat"] == "gate_up_proj"
        if prev_is_gate and not next_is_gate:
            resolved[i] = "down_proj"
        elif next_is_gate and not prev_is_gate:
            resolved[i] = "out_proj"
    return resolved


def band_for_key(key_shape_ms, cat, gx, bx):
    if cat in (None, "unmapped", "out_proj_or_down_proj"):
        return None
    ms = key_shape_ms.get((gx, bx), {}).get(cat)
    if not ms:
        return None
    rep_m = ms[len(ms) // 2]
    return band_for_m(rep_m)


def main():
    labels, key_shape_ms = build_shape_labels()
    launches = fetch_launches(CSV_PATH, labels)
    print(f"total launches: {len(launches)}")

    by_stream = {}
    for launch in launches:
        by_stream.setdefault((launch["dev"], launch["gpid"]), []).append(launch)

    resolved_count = {"out_proj": 0, "down_proj": 0, "unresolved": 0}
    for key, seq in by_stream.items():
        seq.sort(key=lambda l: l["start"])
        verdicts = resolve_by_order(seq)
        for launch, verdict in zip(seq, verdicts):
            if launch["cat"] != "out_proj_or_down_proj":
                continue
            if verdict:
                launch["cat"] = verdict
                resolved_count[verdict] += 1
            else:
                resolved_count["unresolved"] += 1

    print("resolution:", resolved_count)

    cat_counts = {}
    for launch in launches:
        cat_counts[launch["cat"]] = cat_counts.get(launch["cat"], 0) + 1
    print("category counts:", cat_counts)

    durs = []
    for launch in launches:
        dur_us = (launch["end"] - launch["start"]) / 1000.0
        band = band_for_key(key_shape_ms, launch["cat"], launch["gx"], launch["bx"])
        durs.append((dur_us, launch["cat"], band))

    all_dur = [d for d, _, _ in durs]
    print(f"duration range: {min(all_dur):.2f} - {max(all_dur):.2f} us")
    print(f"mean: {sum(all_dur)/len(all_dur):.2f}, median: {sorted(all_dur)[len(all_dur)//2]:.2f}")

    # weighted-mean at gate_up_proj, band 4 (M 1024-2048) -- the specific
    # bucket that showed the original regression.
    band4 = [d for d, cat, band in durs if cat == "gate_up_proj" and band == 4]
    if band4:
        print(f"gate_up_proj band4 (M1024-2048): n={len(band4)}, weighted-mean={sum(band4)/len(band4):.2f}us")

    with open("/tmp/cache_aware_c64_durs.json", "w") as f:
        json.dump(durs, f)
    print("wrote /tmp/cache_aware_c64_durs.json")


if __name__ == "__main__":
    main()
