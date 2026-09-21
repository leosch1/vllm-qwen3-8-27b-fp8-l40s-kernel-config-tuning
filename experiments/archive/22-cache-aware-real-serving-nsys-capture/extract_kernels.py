# Run ON the profiling pod itself (via `oc exec -i ... -- python3 - < extract_kernels.py`,
# uploaded with `oc exec -i ... -- bash -c "cat > extract_kernels.py"` since a direct `oc cp`
# of even this small file failed with a tar error, same as 20) -- NOT locally. The source
# sqlite is the multi-GB export produced by `nsys stats --report gputrace` on this pod's
# own locally-written .nsys-rep (found under /tmp/tmp*/  -- no need to `download` the capture
# via nsight_operator.py at all, since the pod already has it on local disk). Only the small
# CSV result is copied out afterwards -- see README's "gotchas" section (shared with 20) for
# why the sqlite itself is never `oc cp`'d off the pod.
import sqlite3, csv

con = sqlite3.connect("/tmp/tmpmzs4dq5f/default_python3_gpu-node-a_9a16f7ab.sqlite")
cur = con.cursor()

GEMM_KERNEL_NAME = "_w8a8_triton_block_scaled_mm"
row = cur.execute("select id from StringIds where value=?", (GEMM_KERNEL_NAME,)).fetchone()
print("string id:", row)
sid = row[0]

rows = cur.execute(
    """select start, end, gridX, blockX, deviceId, globalPid
       from CUPTI_ACTIVITY_KIND_KERNEL where shortName=?
       order by deviceId, globalPid, start""",
    (sid,),
).fetchall()
print("total launches:", len(rows))

with open("/tmp/cache_aware_c64_launches.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["start", "end", "gridX", "blockX", "deviceId", "globalPid"])
    w.writerows(rows)
print("wrote /tmp/cache_aware_c64_launches.csv")
