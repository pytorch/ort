## 1. Install Python, Git and GPU drivers (optional)

You may need to install some additional libraries on Ubuntu Linux. These steps work on a clean install of Ubuntu Desktop 20.04, and should also work on Ubuntu 18.04 and 20.10, and on Ubuntu Server.

```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install python3-venv build-essential python3-dev git-all
```

If you have a CPU with an Intel Integrated Graphics Card, you can install the [Intel Graphics Compute Runtime](https://github.com/intel/compute-runtime) to enable inference on this device. The command for Ubuntu 20.04 is:

> Note: Only execute this command if you do not yet have OpenCL drivers installed.

```
sudo apt-get install intel-opencl-icd
```

See the [documentation](https://github.com/intel/compute-runtime) for other installation methods and instructions for other versions.

## 2. Install the Notebooks

After installing Python 3 and Git, run each step below in a terminal.

## 3. Create a Virtual Environment

Note: If you already installed tort_infer_env and activated the tort_infer_env environment, you can skip to [Step 5](#5-clone-the-repository).

```bash
python3 -m venv tort_infer_env
```

## 4. Activate the Environment

```bash
source tort_infer_env/bin/activate
```

## 5. Clone the Repository

> Note: Using the `--depth=1` option for git clone reduces download size.

```bash
git clone --depth=1 https://github.com/intel-staging/ort.git
cd ort/torch_ort_inference/demos/notebooks
```

## 6. Install the Packages

This step installs torch-ort-infer python package and dependencies like OpenVINO, Jupyter Lab. First, upgrade pip to the latest version. Then, install the required dependencies. 

```bash
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

## 7. Launch the Notebooks!

If you wish to launch only one notebook, run the command below.

```bash
jupyter "notebook path"
```

To launch all notebooks in Jupyter Lab

```bash
jupyter lab ort/torch_ort_inference/demos/notebooks
```

In Jupyter Lab, select a notebook from the file browser using the left sidebar. Each notebook is located in a subdirectory within the `notebooks` directory.
