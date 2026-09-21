#!/usr/bin/env python3
"""Compare two opt-in W8A8 logical call-plan recordings."""

import argparse
import json
from collections import Counter
from pathlib import Path


def load_plan(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def shape(record: dict[str, object]) -> tuple[int, int, int]:
    return (int(record["M"]), int(record["N"]), int(record["K"]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ordered W8A8 call plans by their logical M/N/K shape."
    )
    parser.add_argument("default_plan", type=Path)
    parser.add_argument("tuned_plan", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    default = load_plan(args.default_plan)
    tuned = load_plan(args.tuned_plan)
    default_counts = Counter(map(shape, default))
    tuned_counts = Counter(map(shape, tuned))
    all_shapes = sorted(set(default_counts) | set(tuned_counts))
    groups = [
        {
            "M": m,
            "N": n,
            "K": k,
            "default_calls": default_counts[(m, n, k)],
            "tuned_calls": tuned_counts[(m, n, k)],
            "pairable_calls": min(
                default_counts[(m, n, k)], tuned_counts[(m, n, k)]
            ),
        }
        for m, n, k in all_shapes
    ]
    same_sequence_positions = sum(
        default_record_shape == tuned_record_shape
        for default_record_shape, tuned_record_shape in zip(
            map(shape, default), map(shape, tuned)
        )
    )
    result = {
        "default_calls": len(default),
        "tuned_calls": len(tuned),
        "shared_shapes": sum(
            count > 0 and tuned_counts[key] > 0
            for key, count in default_counts.items()
        ),
        "default_only_shapes": sum(
            count > 0 and tuned_counts[key] == 0
            for key, count in default_counts.items()
        ),
        "tuned_only_shapes": sum(
            count > 0 and default_counts[key] == 0
            for key, count in tuned_counts.items()
        ),
        "pairable_calls_by_shape": sum(group["pairable_calls"] for group in groups),
        "same_shape_sequence_positions": same_sequence_positions,
        "compared_sequence_positions": min(len(default), len(tuned)),
        "groups": groups,
    }
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
