#!/usr/bin/env python3
"""
Resolve the out_proj vs down_proj ambiguity left open by
../08-per-layer-shape-mapping/ using real vLLM program order, instead of (or
alongside) that experiment's duration-histogram clustering.

Why this is needed: `out_proj` (N=5120, K=3072) and `down_proj` (N=5120,
K=8704) share N, and their tuned/default tile formulas frequently produce the
byte-identical launch grid/block signature -- confirmed directly, ~48-50% of
every real captured GEMM launch in every one of the 4 reliable-instance
captures falls into this exact collision. 08 resolved most of it via
duration-histogram bimodality but explicitly left a residual ~3-3.5%
genuinely ambiguous even after that.

The trick here is structural, not statistical: within one decoder layer, the
real forward pass issues kernels in a fixed program order --
    {qkv_proj | in_proj_qkvz} -> attention -> out_proj -> gate_up_proj -> down_proj
So the ambiguous 5120-shape launch immediately BEFORE a gate_up_proj launch
is that layer's out_proj; the one immediately AFTER is that layer's
down_proj. gate_up_proj itself is never ambiguous (unique grid divisor), so
it's a reliable anchor.

This only works if launch order == program order, i.e. no concurrent-stream
interleaving scrambles the timeline. Confirmed directly: every capture used
here has exactly 2 distinct (deviceId, globalPid, streamId, contextId)
combinations -- one CUDA stream per device (TP=2) -- so sorting by `start`
within one device reproduces true issue order.

No live cluster/Nsight Operator access needed: every input already exists
locally as `kernel_trace.sqlite` under
../05-nsys-hardware-profiling-matrix/profiling-results/*-v2/, extracted
during that experiment. This script is pure offline reanalysis of that data.

Run from the repo root:
    python3 experiments/18-out-down-proj-order-disambiguation/resolve_order.py
"""

import json
import math
import sqlite3

TUNED_DIR = "./tuned-configs"
PROFILING_BASE = "./experiments/05-nsys-hardware-profiling-matrix/profiling-results"
GEMM_KERNEL_NAME = "_w8a8_triton_block_scaled_mm"

SHAPES = {
    "gate_up_proj": (17408, 5120),
    "in_proj_qkvz": (8192, 5120),
    "qkv_proj": (7168, 5120),
    "out_proj": (5120, 3072),
    "down_proj": (5120, 8704),
}

CAPTURES = [
    ("default c64", "pass1-default-concurrency64-v2", "default"),
    ("tuned c64", "pass2-tuned-concurrency64-v2", "tuned"),
    ("default c1", "pass1-default-concurrency1-v2", "default"),
    ("tuned c1", "pass2-tuned-concurrency1-v2", "tuned"),
]


def _load_tuned_config(n, k):
    fn = (
        f"{TUNED_DIR}/N={n},K={k},device_name=NVIDIA_L40S,"
        f"dtype=fp8_w8a8,block_shape=[128,128].json"
    )
    return {int(m): v for m, v in json.load(open(fn)).items()}


def build_shape_labels(max_m=4096):
    """(gridX -> label) for default's fixed formula, ((gridX,blockX) -> label) for
    tuned's per-anchor lookup. Exactly 08-per-layer-shape-mapping's formula.
    A gridX/gridX+blockX achievable by exactly one shape gets that shape's name;
    achievable by exactly {out_proj, down_proj} gets "out_proj_or_down_proj";
    anything else (no formula match, or a rarer 3+-way collision) -> "unmapped".
    """
    tuned_cfgs = {name: _load_tuned_config(n, k) for name, (n, k) in SHAPES.items()}

    def nearest_anchor(anchors, m):
        return min(anchors, key=lambda a: abs(a - m))

    def default_gridx(n, m):
        return math.ceil(m / 64) * math.ceil(n / 128)

    def tuned_gridx_blockx(name, m):
        n, k = SHAPES[name]
        cfg_by_anchor = tuned_cfgs[name]
        anchors = sorted(cfg_by_anchor)
        cfg = cfg_by_anchor[nearest_anchor(anchors, m)]
        gx = math.ceil(m / cfg["BLOCK_SIZE_M"]) * math.ceil(n / cfg["BLOCK_SIZE_N"])
        return gx, cfg["num_warps"] * 32

    pair = frozenset({"out_proj", "down_proj"})

    def label_for(shape_set):
        if len(shape_set) == 1:
            return next(iter(shape_set))
        return "out_proj_or_down_proj" if frozenset(shape_set) == pair else "unmapped"

    default_raw, tuned_raw = {}, {}
    for name, (n, _k) in SHAPES.items():
        for m in range(1, max_m + 1):
            default_raw.setdefault(default_gridx(n, m), set()).add(name)
            tuned_raw.setdefault(tuned_gridx_blockx(name, m), set()).add(name)

    default_labels = {gx: label_for(s) for gx, s in default_raw.items()}
    tuned_labels = {key: label_for(s) for key, s in tuned_raw.items()}
    return default_labels, tuned_labels


def classify(default_labels, tuned_labels, config, grid_x, block_x):
    if config == "default":
        return default_labels.get(grid_x, "unmapped")
    return tuned_labels.get((grid_x, block_x), "unmapped")


def fetch_launches(folder, default_labels, tuned_labels, config):
    con = sqlite3.connect(f"{PROFILING_BASE}/{folder}/kernel_trace.sqlite")
    cur = con.cursor()
    (sid,) = cur.execute(
        "select id from StringIds where value=?", (GEMM_KERNEL_NAME,)
    ).fetchone()
    rows = cur.execute(
        """select start, end, gridX, blockX, deviceId, globalPid
           from CUPTI_ACTIVITY_KIND_KERNEL where shortName=?
           order by deviceId, globalPid, start""",
        (sid,),
    ).fetchall()
    con.close()
    return [
        {
            "start": start,
            "end": end,
            "dev": dev,
            "cat": classify(default_labels, tuned_labels, config, gx, bx),
        }
        for start, end, gx, bx, dev, _gpid in rows
    ]


def resolve_by_order(seq):
    """seq must already be one device's launches, sorted by start time.
    Returns a list parallel to seq: 'out_proj' / 'down_proj' / None (still
    ambiguous) for each launch originally labeled 'out_proj_or_down_proj'."""
    n = len(seq)
    resolved = [None] * n
    for i, launch in enumerate(seq):
        if launch["cat"] != "out_proj_or_down_proj":
            continue
        prev_is_gate = i > 0 and seq[i - 1]["cat"] == "gate_up_proj"
        next_is_gate = i < n - 1 and seq[i + 1]["cat"] == "gate_up_proj"
        if prev_is_gate and not next_is_gate:
            resolved[i] = "down_proj"  # right after gate_up_proj
        elif next_is_gate and not prev_is_gate:
            resolved[i] = "out_proj"  # right before gate_up_proj
        # else: both or neither neighbor is gate_up_proj -> leave ambiguous
    return resolved


def analyze_capture(label, folder, config, default_labels, tuned_labels):
    launches = fetch_launches(folder, default_labels, tuned_labels, config)
    n_total = len(launches)
    n_gate = sum(1 for launch in launches if launch["cat"] == "gate_up_proj")
    n_ambiguous = sum(
        1 for launch in launches if launch["cat"] == "out_proj_or_down_proj"
    )
    n_already_out = sum(1 for launch in launches if launch["cat"] == "out_proj")
    n_already_down = sum(1 for launch in launches if launch["cat"] == "down_proj")

    resolved_out = resolved_down = unresolved = 0
    durations_out, durations_down = [], []
    for dev in sorted({launch["dev"] for launch in launches}):
        seq = sorted(
            (launch for launch in launches if launch["dev"] == dev),
            key=lambda launch: launch["start"],
        )
        for launch, verdict in zip(seq, resolve_by_order(seq)):
            if launch["cat"] != "out_proj_or_down_proj":
                continue
            dur_us = (launch["end"] - launch["start"]) / 1000.0
            if verdict == "out_proj":
                resolved_out += 1
                durations_out.append(dur_us)
            elif verdict == "down_proj":
                resolved_down += 1
                durations_down.append(dur_us)
            else:
                unresolved += 1

    total_out = n_already_out + resolved_out
    total_down = n_already_down + resolved_down
    mean_out = sum(durations_out) / len(durations_out) if durations_out else None
    mean_down = sum(durations_down) / len(durations_down) if durations_down else None

    return {
        "label": label,
        "n_total": n_total,
        "n_gate_up_proj": n_gate,
        "n_ambiguous": n_ambiguous,
        "ambiguous_pct_of_total": 100 * n_ambiguous / n_total,
        "resolved_out_proj": resolved_out,
        "resolved_down_proj": resolved_down,
        "unresolved": unresolved,
        "unresolved_pct_of_ambiguous": 100 * unresolved / n_ambiguous,
        "total_out_proj": total_out,
        "total_down_proj": total_down,
        "out_proj_over_gate_up_proj": total_out / n_gate,
        "down_proj_over_gate_up_proj": total_down / n_gate,
        "mean_us_resolved_out_proj": mean_out,
        "mean_us_resolved_down_proj": mean_down,
        "duration_ratio_down_over_out": (
            mean_down / mean_out if mean_out and mean_down else None
        ),
    }


def main():
    default_labels, tuned_labels = build_shape_labels()
    results = [
        analyze_capture(label, folder, config, default_labels, tuned_labels)
        for label, folder, config in CAPTURES
    ]
    for r in results:
        print(f"=== {r['label']} ===")
        print(f"  total GEMM launches: {r['n_total']:,}")
        print(f"  gate_up_proj (anchor): {r['n_gate_up_proj']:,}")
        print(
            f"  ambiguous pair before resolution: {r['n_ambiguous']:,} "
            f"({r['ambiguous_pct_of_total']:.2f}% of all launches)"
        )
        print(
            f"  resolved via ordering: out_proj +{r['resolved_out_proj']:,}, "
            f"down_proj +{r['resolved_down_proj']:,}, "
            f"still unresolved: {r['unresolved']:,} "
            f"({r['unresolved_pct_of_ambiguous']:.3f}% of ambiguous)"
        )
        print(
            f"  FINAL: out_proj={r['total_out_proj']:,}, "
            f"down_proj={r['total_down_proj']:,}"
        )
        print(
            f"  sanity (expect ~1.0, 1-per-layer architecture): "
            f"out_proj/gate_up_proj={r['out_proj_over_gate_up_proj']:.4f}, "
            f"down_proj/gate_up_proj={r['down_proj_over_gate_up_proj']:.4f}"
        )
        if r["duration_ratio_down_over_out"]:
            print(
                f"  cross-check: resolved-out_proj mean={r['mean_us_resolved_out_proj']:.1f}us, "
                f"resolved-down_proj mean={r['mean_us_resolved_down_proj']:.1f}us, "
                f"ratio={r['duration_ratio_down_over_out']:.2f} "
                f"(K ratio 8704/3072={8704/3072:.2f})"
            )
        print()

    json.dump(
        results,
        open(
            "experiments/18-out-down-proj-order-disambiguation/results.json", "w"
        ),
        indent=2,
    )


if __name__ == "__main__":
    main()
