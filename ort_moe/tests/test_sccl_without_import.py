# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os

os.environ["LD_LIBRARY_PATH"] = "/usr/src/nccl-master-0.3.1/build/lib/"

# run sccl without import sccl -- should be identical to nccl 2.8.4
os.system("/usr/src/nccl-tests/build/alltoall_perf -b 128 -e 1GB -f 2 -g 1 -c 1 -n 200 -w 10 -z 0")
