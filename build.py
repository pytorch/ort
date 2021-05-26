import argparse
import os
import sys
import subprocess

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--wheel_file", help="wheel filename used to test. skip build wheel and install")
    return parser.parse_args()

def run_subprocess(args, cwd=None):
    if isinstance(args, str):
        raise ValueError("args should be a sequence of strings, not a string")

    return subprocess.run(args, cwd=cwd, shell=False, check=True)

def run_ort_module_tests(cwd, source_dir):
    args = [sys.executable, os.path.join(source_dir, 'tests/bert_for_sequence_classification.py')]
    run_subprocess(args, cwd)

def build_wheel(cwd, source_dir, cmd_line_args):
    args = [sys.executable, os.path.join(source_dir, 'setup.py'), 'bdist_wheel']
    
    if cmd_line_args.version_str:
        # to override default version_str in version.txt
        args.append('--version_str={}'.format(cmd_line_args.version_str))

    run_subprocess(args, cwd)

def main():
    cmd_line_args = parse_arguments()

    source_dir = os.path.realpath(os.path.dirname(__file__))
    cwd = os.path.normpath(os.path.join(source_dir, ".."))

    if not cmd_line_args.wheel_file:
        build_wheel(source_dir, source_dir, cmd_line_args)
        # installing torch-ort wheel
        dist_path = os.path.join(source_dir, 'dist')
        wheel_file = os.listdir(dist_path)[0]
        run_subprocess([sys.executable, "-m", "pip", "install", "--upgrade", os.path.join(dist_path, wheel_file)], cwd)
    else:
        print("cmd_line_args.wheel_file:", cmd_line_args.wheel_file)
        run_subprocess([sys.executable, "-m", "pip", "install", "--upgrade", cmd_line_args.wheel_file], cwd)

    # installing requirements-test.txt
    requirements_path = os.path.join(source_dir, 'tests', 'requirements-test.txt')
    run_subprocess([sys.executable, "-m", "pip", "install", "-r", requirements_path], cwd)

    # testing torch-ort
    run_ort_module_tests(source_dir, source_dir)

    # upload wheel. ask for conformation if it is for stable build - nightly build can be uploaded from pipelines.
    # release/stable build can only be uploaded manually

if __name__ == "__main__":
    sys.exit(main())
