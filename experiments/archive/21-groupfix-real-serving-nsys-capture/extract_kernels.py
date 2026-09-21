# Run ON the profiling pod itself (via `oc exec -i ... -- python3 - < extract_kernels.py`,
# uploaded with `oc exec -i ... -- bash -c "cat > extract_kernels.py"` since a direct `oc cp`
# of even this small file failed with a tar error) -- NOT locally. The source sqlite is the
# multi-GB export produced by `nsys stats --report gputrace` (see README's "gotchas" section
# for why `nsys export --type sqlite` itself is not used and why the sqlite is never `oc cp`'d
# off the pod). Only the small CSV result is copied out afterwards.
import sqlite3, csv

con = sqlite3.connect("/tmp/tmpgqoq1gvu/default_python3_gpu-node-a_2a107cae.sqlite")
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

with open("/tmp/groupfix_c64_launches.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["start", "end", "gridX", "blockX", "deviceId", "globalPid"])
    w.writerows(rows)
print("wrote /tmp/groupfix_c64_launches.csv")
