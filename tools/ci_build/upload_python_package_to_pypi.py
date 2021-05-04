#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import argparse
import subprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload python whl to azure storage.")

    parser.add_argument("--python_wheel_path", type=str, help="path to python wheel")
    parser.add_argument("--account_name", type=str, help="account name")
    parser.add_argument("--account_token", type=str, help="account token")

    # TODO: figure out a way to secure args.account_token to prevent later code changes
    # that may accidentally print out it to the console.
    args = parser.parse_args()

    subprocess.run([
        "twine",
        "upload",
        "--username",
        args.account_name,
        "--password", 
        args.account_token,
        args.python_wheel_path])

    
