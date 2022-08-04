
#  ResNet-50 Image Classification

This demo shows how to use Intel® OpenVINO™ integration with Torch-ORT to classify objects in images with ONNX Runtime OpenVINO Execution Provider.

We use an image classification model [ResNet-50](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50) from Torchvision and [ImageNet labels](https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt) to classify objects. In the labels file, you'll find 1,000 different categories that were used in the Imagenet competition.
  

## Data
The input to the model is a 224 x 224 image (airplane in our case), and the **output** is a list of estimated class probabilities.

<p align="center" width="100%"> <img src="plane.jpg" alt="drawing" height="300" width="400"/>

## Model Metadata
| Domain | Application | Industry  | Framework | Training Data | Input Data Format |
| ------------- | --------  | -------- | --------- | --------- | -------------- | 
| Vision | Image Classification | General | Pytorch | [ImageNet](http://www.image-net.org/) | Image (RGB/HWC)|

## Pre-requisites

- Ubuntu 18.04, 20.04

- Python* 3.7, 3.8 or 3.9

## Install in a local Python environment

1. Upgrade pip

    - `pip install --upgrade pip`
<br/><br/>

2. Install torch-ort-infer with OpenVINO dependencies

    - `pip install torch-ort-infer[openvino]`
<br/><br/>
3. Run post-installation script

    - `python -m torch_ort.configure`

## Verify your installation

Once you have created your environment, execute the following steps to validate that your installation is correct.

1. Clone this repo

    - `git clone https://github.com/pytorch/ort.git`
<br/><br/>
2. Install extra dependencies

    - `pip install wget Pillow torchvision`
<br/><br/>
3. Run the inference script

    - `python ./ort/torch_ort_inference/demos/resnet_image_classification.py --input-file ./ort/torch_ort_inference/demos/plane.jpg`
<br/><br/>
    **Note**: OpenVINOExecutionProvider is enabled with CPU and FP32 by default.
<br/><br/>
4. To see more options for various backends (Intel® hardware), invoke:

    - `$ python ./ort/torch_ort_inference/demos/resnet_image_classification.py --help`

## Output

For the input image of an airplane, you can see the output something similar to:

```
Labels , Probabilities:
airliner 0.9133861660957336
wing 0.08387967199087143
airship 0.001151240081526339
warplane 0.00030989135848358274
projectile 0.0002502237621229142

```

Here, the network classifies the image as an airplane, with a high score of 0.91.

## Run inference using new input
You can provide either absolute or relative path to an image of your choice.
e.g.
```bash
$ python ./ort/torch_ort_inference/demos/resnet_image_classification.py  --input-file=<absolute-or-relative-path-to-your-input>
```
If you add a new image (e.g, my_image.png) to the existing `<path-to-ort-repository>/torch_ort_inference/demos` directory in the **ort** repository, it will look like this:

```bash
$ python ./ort/torch_ort_inference/demos/resnet_image_classification.py --input-file=./ort/torch_ort_inference/demos/my_image.png

```
<br/> 

**Note**: This demo has a warm-up run and then inference time is measured on the subsequent runs. The execution time of first run is in general higher compared to the next runs as it includes inline conversion to ONNX, many one-time graph transformations and optimizations steps.

For more details on APIs, see [usage.md](/torch_ort_inference/docs/usage.md).



