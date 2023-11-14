# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os

os.environ["LD_LIBRARY_PATH"] = "/usr/src/nccl-2.8.4-1/build/lib/"

# run original nccl 2.8.4 baseline
os.system("/usr/src/nccl-tests-baseline/build/alltoall_perf -b 128 -e 1GB -f 2 -g 1 -c 1 -n 200 -w 10 -z 0")
