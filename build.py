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

def run_ort_module_tests(cwd, source_dir):
    args = [sys.executable, os.path.join(source_dir, 'tests/bert_for_sequence_classification.py')]
    run_subprocess(args, cwd)

def build_wheel(cwd, source_dir):
    args = [sys.executable, os.path.join(source_dir, 'setup.py'), 'bdist_wheel']
    run_subprocess(args, cwd)

def main():    
    source_dir = os.path.realpath(os.path.dirname(__file__))
    cwd = os.path.normpath(os.path.join(source_dir, ".."))
    run_subprocess([sys.executable, "-m", "pip", "list"], cwd)

    build_wheel(source_dir, source_dir)
        
    dist_path = os.path.join(source_dir, 'dist')
    wheel_file = os.listdir(dist_path)[0]
    run_subprocess([sys.executable, "-m", "pip", "install", "--upgrade", os.path.join(dist_path, wheel_file)], cwd)
    run_subprocess([sys.executable, "-m", "pip", "list"], cwd)

    run_ort_module_tests(cwd, source_dir)

if __name__ == "__main__":
    sys.exit(main())
