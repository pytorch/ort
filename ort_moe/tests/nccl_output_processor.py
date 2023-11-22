import os,re,sys

if len(sys.argv) != 3:
    print("usage: python nccl_output_processor.py nccl_output sccl_output")
file1 = open(sys.argv[1],"r")
file2 = open(sys.argv[2],"r")
nccl_res = []
for l in file1.readlines():
    g = re.match("\s+(\d+)\s+\d+\s+float\s+([0-9]*\.?[0-9]+)\s+[0-9]*\.?[0-9]+\s+[0-9]*\.?[0-9]+\s+(\d+e\+\d+)", l)
    if g is not None:
        nccl_res.append((float(g.group(1)), float(g.group(2)), float(g.group(3))))
# the sccl output
sccl_res = []
for l in file2.readlines():
    g = re.match("\s+(\d+)\s+\d+\s+float\s+([0-9]*\.?[0-9]+)\s+[0-9]*\.?[0-9]+\s+[0-9]*\.?[0-9]+\s+(\d+e\+\d+)", l)
    if g is not None:
        sccl_res.append((float(g.group(1)), float(g.group(2)), float(g.group(3))))
counter = 0
for a,b in zip(nccl_res, sccl_res):
    if a[0] != b[0]:
        print("Sizes didn't match in sccl/nccl comparison")
        exit(-1)
    # Make sure SCCL is not more than 10% slower than NCCL. Always skip the first size as it is unstable.
    if b[1] > a[1]*1.05 and counter > 0:
        print(f"Performance of sccl slowed down for size {a[0]}: nccl {a[1]} vs sccl {b[1]}")
        exit(-1)
    if a[2] > 0 or b[2] > 0:
        print(f"Correctness did not pass for size {a[0]}: nccl {a[2]}, sccl {b[2]}")
        exit(-1)
    counter += 1
print("All checks passed!")
