# Run ON the profiling pod itself (uploaded with
# `oc exec -i ... -- bash -c "cat > extract_kernels.py"`, same as 20/21) --
# NOT locally. The source sqlite is the multi-GB export produced by
# `nsys stats --report gputrace` on this pod's own locally-written
# .nsys-rep. Note: the first attempt here hit "Exportation error: Section
# Table Reference magic number mismatch" -- the .nsys-rep was still being
# finalized/flushed to disk when nsys stats first ran against it (file size
# grew between the profiler-stop and the first extraction attempt).
# Waiting ~15s for the file size to stabilize before retrying resolved it
# cleanly -- a new gotcha not seen in 20/21, worth checking file size
# stability before running `nsys stats` on freshly-stopped captures.
# Only the small CSV result is copied out afterwards -- see 20's README for
# why the sqlite itself is never `oc cp`'d off the pod.
import sqlite3, csv

con = sqlite3.connect("/tmp/tmpqr0s9rtl/default_python3_gpu-node-a_51662443.sqlite")
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

with open("/tmp/singlefix_c64_launches.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["start", "end", "gridX", "blockX", "deviceId", "globalPid"])
    w.writerows(rows)
print("wrote /tmp/singlefix_c64_launches.csv")
