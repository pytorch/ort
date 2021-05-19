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

    print("installing requirements-test.txt")
    requirements_path = os.path.join(source_dir, 'tests', 'requirements-test.txt')
    run_subprocess([sys.executable, "-m", "pip", "install", "-r", requirements_path], cwd)

    build_wheel(source_dir, source_dir)

    print("installing torch-ort wheel")
    dist_path = os.path.join(source_dir, 'dist')
    wheel_file = os.listdir(dist_path)[0]
    run_subprocess([sys.executable, "-m", "pip", "install", "--upgrade", os.path.join(dist_path, wheel_file)], cwd)

    print("testing torch-ort")
    run_ort_module_tests(source_dir, source_dir)

if __name__ == "__main__":
    sys.exit(main())
