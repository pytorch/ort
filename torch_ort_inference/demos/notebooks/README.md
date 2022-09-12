<h1 align="center">📚 Openvino Integration with Torch-ORT Notebooks</h1>

[![MIT License](https://img.shields.io/apm/l/vim-mode)](https://github.com/pytorch/ort/blob/main/LICENSE)

A collection of ready-to-run Jupyter notebooks for learning and experimenting with the torch-ort-infer python package. The notebooks provide an introduction to torch-ort-infer basics and teach developers how to leverage our API for optimized deep learning inference.

To upgrade to the new release version, please run `pip install --upgrade -r requirements.txt` in your `tort_infer_env` virtual environment. If you need to install for the first time, see the [Installation Guide](#-installation-guide) section below.

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()

## Table of Contents

* [➤ 📝 Installation Guide](#-installation-guide)
	* [Ubuntu](../notebooks/UBUNTU.md)
* [➤ ⚙️ System Requirements](#-system-requirements)
* [➤ 💻 Run the Notebooks](#-run-the-notebooks)
* [➤ 🧹 Cleaning Up](#-cleaning-up)
* [➤ 🧑‍💻 Contributors](#-contributors)

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()
<div id='-installation-guide'/>

## 📝 Installation Guide

| [Ubuntu](../notebooks/UBUNTU.md) | 
| ------------------------------------------------------------------------------------------ | 


## ⚙️ System Requirements

The table below lists the supported operating systems and Python versions. **Note:** Python 3.10 is not supported yet.

| Supported Operating System                                 | [Python Version (64-bit)](https://www.python.org/) |
| :--------------------------------------------------------- | :------------------------------------------------- |
| Ubuntu 18.04 LTS, 64-bit                                 | 3.6, 3.7, 3.8, 3.9                                      |
| Ubuntu 20.04 LTS, 64-bit                                 | 3.6, 3.7, 3.8, 3.9                                      |

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)](#)
<div id='-run-the-notebooks'/>

## 💻 Run the Notebooks

### To Launch a Single Notebook

If you wish to launch only one notebook, run the command below.

```bash
jupyter "notebook path"
```

### To Launch all Notebooks

```bash
jupyter lab notebooks
```

In your browser, select a notebook from the file browser in Jupyter Lab using the left sidebar. Each tutorial is located in a subdirectory within the `notebooks` directory.

<img src="https://user-images.githubusercontent.com/15709723/120527271-006fd200-c38f-11eb-9935-2d36d50bab9f.gif">

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()

<div id='-cleaning-up'/>

## 🧹 Cleaning Up

<p>
<details>
<summary>Shut Down Jupyter Kernel</summary>

To end your Jupyter session, press `Ctrl-c`. This will prompt you to `Shutdown this Jupyter server (y/[n])?` enter `y` and hit `Enter`.
</details>
</p>	
	
<p>
<details>
<summary>Deactivate Virtual Environment</summary>

To deactivate your virtualenv, simply run `deactivate` from the terminal window where you activated `tort_infer_env`. This will deactivate your environment.

To reactivate your environment, run `source tort_infer_env/bin/activate` on Linux or `tort_infer_env\Scripts\activate` on Windows, then type `jupyter lab` or `jupyter notebook` to launch the notebooks again.
</details>
</p>	
	
<p>
<details>
<summary>Delete Virtual Environment _(Optional)_</summary>

To remove your virtual environment, simply delete the `tort_infer_env` directory:
</details>
</p>	
	
<p>
<details>
<summary>On Linux and macOS:</summary>

```bash
rm -rf tort_infer_env
```
</details>
</p>

<p>
<details>
<summary>On Windows:</summary>

```bash
rmdir /s tort_infer_env
```
</details>
</p>

<p>
<details>
<summary>Remove tort_infer_env Kernel from Jupyter</summary>

```bash
jupyter kernelspec remove tort_infer_env
```
</details>
</p>
