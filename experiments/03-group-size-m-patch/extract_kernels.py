#!/usr/bin/env python3
"""
Run ON the profiling pod, against the sqlite `nsys stats --report gputrace`
produces from a captured .nsys-rep (see ../02-nsys-shape-profiling/README.md
for the capture prerequisites -- DCGM paused, --cuda-graph-trace=node).

Pulls every _w8a8_triton_block_scaled_mm launch's raw (start, end, gridX,
blockX) into a CSV. gridX (thread-block grid size) plus blockX (threads per
block) is enough to work out which of this project's known launch-shape
signatures a launch matches, since GROUP_SIZE_M -- the only thing the
singlefix config changes -- doesn't affect grid or block dimensions.
Deriving (N, K, M) from a signature match is the next, config-specific step
(see this folder's README); this script only extracts the raw rows.

Usage: python3 extract_kernels.py <path-to-nsys-export.sqlite> <output.csv>
"""

import csv
import sqlite3
import sys

GEMM_KERNEL_NAME = "_w8a8_triton_block_scaled_mm"


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <export.sqlite> <output.csv>", file=sys.stderr)
        sys.exit(1)
    sqlite_path, out_csv = sys.argv[1], sys.argv[2]

    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()

    row = cur.execute(
        "select id from StringIds where value=?", (GEMM_KERNEL_NAME,)
    ).fetchone()
    if row is None:
        print(f"'{GEMM_KERNEL_NAME}' not found in this capture's StringIds", file=sys.stderr)
        sys.exit(1)
    sid = row[0]

    rows = cur.execute(
        """select start, end, gridX, blockX, deviceId, globalPid
           from CUPTI_ACTIVITY_KIND_KERNEL where shortName=?
           order by deviceId, globalPid, start""",
        (sid,),
    ).fetchall()
    print(f"total launches: {len(rows)}")

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start", "end", "gridX", "blockX", "deviceId", "globalPid"])
        w.writerows(rows)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
