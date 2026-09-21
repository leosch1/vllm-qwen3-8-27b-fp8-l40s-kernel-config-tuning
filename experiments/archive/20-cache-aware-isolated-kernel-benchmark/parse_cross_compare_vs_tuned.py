#!/usr/bin/env python3
"""
Cross-compare cache-aware and groupfix against the *original tuned* config
(not default), under cache-cycling pressure -- reusing the raw absolute
per-M duration numbers already logged in 19's three cycling runs
(compare_default_vs_cache_aware_cycling.py, compare_default_vs_groupfix_cycling.py,
compare_default_vs_tuned_cycling.py). All three used the identical cycling
design (64 B-copies, counter reset per candidate, same M anchor/held-out
list, same NUM_ITERS, same hardware) -- so their absolute microsecond
durations are directly comparable to each other without needing to chain
through each run's own `default` baseline.
"""
import re
import json

DIR = "."

SHAPE_ORDER = ["gate_up_proj", "in_proj_qkvz", "qkv_proj", "down_proj", "out_proj"]
SHAPE_NK = {
    "gate_up_proj": (17408, 5120),
    "in_proj_qkvz": (8192, 5120),
    "qkv_proj": (7168, 5120),
    "down_proj": (5120, 8704),
    "out_proj": (5120, 3072),
}

ROW_RE = re.compile(
    r"\|\s*(\d+)\s*\|\s*(anchor|held-out)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([+-][\d.]+)%\s*\|"
)


def parse_log(path):
    """Returns {shape_key: {M: {"type":..., "anchor":..., "default": float, "candidate": float}}}"""
    with open(path) as f:
        lines = f.readlines()
    shapes_seen = []
    result = {}
    cur_shape_idx = -1
    for line in lines:
        if line.startswith("### N="):
            cur_shape_idx += 1
            shapes_seen.append(SHAPE_ORDER[cur_shape_idx])
            result[SHAPE_ORDER[cur_shape_idx]] = {}
            continue
        m = ROW_RE.match(line.strip())
        if m:
            M, typ, anchor, default_us, cand_us, _speedup = m.groups()
            result[SHAPE_ORDER[cur_shape_idx]][int(M)] = {
                "type": typ,
                "anchor": int(anchor),
                "default": float(default_us),
                "candidate": float(cand_us),
            }
    return result


def main():
    cache_aware = parse_log(f"{DIR}/isolated-benchmark-cycling-results.log")
    tuned = parse_log(f"{DIR}/isolated-benchmark-tuned-cycling-results.log")
    groupfix = parse_log(f"{DIR}/isolated-benchmark-groupfix-cycling-results.log")

    # sanity check: how close are the three runs' own `default` numbers to
    # each other? (they should be very close -- same fixed config/script/hw)
    print("=== sanity: default-vs-default spread across the 3 runs (gate_up_proj) ===")
    for M in sorted(cache_aware["gate_up_proj"]):
        d_ca = cache_aware["gate_up_proj"][M]["default"]
        d_tu = tuned["gate_up_proj"][M]["default"]
        d_gf = groupfix["gate_up_proj"][M]["default"]
        spread = (max(d_ca, d_tu, d_gf) - min(d_ca, d_tu, d_gf)) / min(d_ca, d_tu, d_gf) * 100
        if spread > 5:
            print(f"  M={M}: default_ca={d_ca} default_tuned={d_tu} default_groupfix={d_gf} spread={spread:.1f}%")
    print("(rows only printed if spread > 5% -- silence means all runs' defaults track tightly)")
    print()

    def cross_compare(name, cand_data, out_path):
        results = {}
        neg = 0
        total = 0
        worst = (None, None, 0)
        for shape in SHAPE_ORDER:
            results[shape] = []
            for M in sorted(cand_data[shape]):
                if M not in tuned[shape]:
                    continue
                cand_us = cand_data[shape][M]["candidate"]
                tuned_us = tuned[shape][M]["candidate"]
                typ = cand_data[shape][M]["type"]
                anchor = cand_data[shape][M]["anchor"]
                speedup_pct = (tuned_us - cand_us) / tuned_us * 100
                results[shape].append({
                    "M": M, "type": typ, "anchor": anchor,
                    f"{name}_us": cand_us, "tuned_us": tuned_us,
                    "speedup_pct": round(speedup_pct, 1),
                })
                total += 1
                if speedup_pct < 0:
                    neg += 1
                if speedup_pct < worst[2]:
                    worst = (shape, M, speedup_pct)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"=== {name} vs tuned (cycling): {neg}/{total} negative, worst={worst}")
        return results

    cache_aware_vs_tuned = cross_compare("cache_aware", cache_aware, "cache_aware_vs_tuned_cycling.json")
    groupfix_vs_tuned = cross_compare("groupfix", groupfix, "groupfix_vs_tuned_cycling.json")


if __name__ == "__main__":
    main()
