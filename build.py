import argparse
import os
import sys
import subprocess

def parse_arguments():
    parser = argparse.ArgumentParser()

def run_subprocess(args, cwd=None, capture_stdout=False, shell=False):
    if isinstance(args, str):
        raise ValueError("args should be a sequence of strings, not a string")

    return subprocess.run(args, cwd=cwd, shell=shell)

def run_ort_module_tests(source_dir):
    args = [sys.executable, os.path.join(source_dir, 'tests/bert_for_sequence_classification.py')]
    run_subprocess(args, source_dir)

def build_wheel(source_dir):
    args = [sys.executable, os.path.join(source_dir, 'setup.py'), 'bdist_wheel']
    run_subprocess(args, source_dir)

def main():
    source_dir = os.path.realpath(os.path.dirname(__file__))
    run_ort_module_tests(source_dir)
    build_wheel(source_dir)

if __name__ == "__main__":
    sys.exit(main())
