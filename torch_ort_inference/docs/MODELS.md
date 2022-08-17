This document provides a list of all validated models that are supported by **OpenVINO™ integration with Torch-ORT**. The list of supported models and performance is continuously evolving as we are optimizing models and enabling more operators. The performance gain depends on various factors such as model architecture, Intel<sup>®</sup> Platform (e.g. Xeon<sup>®</sup> or Core<sup>TM</sup>), Backend device (e.g. CPU, GPU or VPU) etc. 


## Torchvision Models
| Model Name | Supported Devices |
|---|---|
| [alexnet](https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py)  | CPU, iGPU |
| [deeplabv3_mobilenet_v3_large](https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py)  | CPU, iGPU |
| [deeplabv3_resnet101](https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py)  | CPU, iGPU |
| [deeplabv3_resnet50](https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py)  | CPU, iGPU |
| [densenet121](https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py)  | CPU, iGPU |
| [densenet161](https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py)  | CPU, iGPU |
| [densenet169](https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py)  | CPU, iGPU |
| [densenet201](https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py)  | CPU, iGPU |
| [efficientnet_b0](https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py)  | CPU, iGPU |
| [efficientnet_b1](https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py)  | CPU, iGPU |
| [efficientnet_b2](https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py)  | CPU, iGPU |
| [efficientnet_b3](https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py)  | CPU, iGPU |
| [efficientnet_b4](https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py)  | CPU, iGPU |
| [efficientnet_b5](https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py)  | CPU, iGPU |
| [efficientnet_b6](https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py)  | CPU, iGPU |
| [efficientnet_b7](https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py)  | CPU, iGPU |
| [fcn_resnet101](https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py)  | CPU, iGPU |
| [fcn_resnet50](https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py)  | CPU, iGPU |
| [googlenet](https://github.com/pytorch/vision/blob/main/torchvision/models/googlenet.py)  | CPU, iGPU |
| [inception_v3](https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py)  | CPU, iGPU |
| [lraspp_mobilenet_v3_large](https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/lraspp.py)  | CPU, iGPU |
| [mnasnet0_5](https://github.com/pytorch/vision/blob/main/torchvision/models/mnasnet.py)  | CPU, iGPU |
| [mnasnet1_0](https://github.com/pytorch/vision/blob/main/torchvision/models/mnasnet.py)  | CPU, iGPU |
| [mobilenet_v2](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py)  | CPU, iGPU |
| [mobilenet_v3_large](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py)  | CPU, iGPU |
| [mobilenet_v3_small](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py)  | CPU, iGPU |
| [regnet_x_1_6gf](https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py)  | CPU, iGPU |
| [regnet_x_16gf](https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py)  | CPU, iGPU |
| [regnet_x_3_2gf](https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py)  | CPU, iGPU |
| [regnet_x_32gf](https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py)  | CPU, iGPU |
| [regnet_x_400mf](https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py)  | CPU, iGPU |
| [regnet_x_800mf](https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py)  | CPU, iGPU |
| [regnet_x_8gf](https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py)  | CPU, iGPU |
| [regnet_y_1_6gf](https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py)  | CPU, iGPU |
| [regnet_y_16gf](https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py)  | CPU, iGPU |
| [regnet_y_3_2gf](https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py)  | CPU, iGPU |
| [regnet_y_400mf](https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py)  | CPU, iGPU |
| [regnet_y_800mf](https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py)  | CPU, iGPU |
| [regnet_y_8gf](https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py)  | CPU, iGPU |
| [resnet_2plus1d](https://github.com/pytorch/vision/blob/main/torchvision/models/video/resnet.py)  | CPU, iGPU |
| [resnet_3d](https://github.com/pytorch/vision/blob/main/torchvision/models/video/resnet.py)  | CPU, iGPU |
| [resnet_mixed_conv_3d](https://github.com/pytorch/vision/blob/main/torchvision/models/video/resnet.py)  | CPU, iGPU |
| [resnet101](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)  | CPU, iGPU |
| [resnet152](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)  | CPU, iGPU |
| [resnet18](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)  | CPU, iGPU |
| [resnet34](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)  | CPU, iGPU |
| [resnet50](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)  | CPU, iGPU |
| [resnext101_32x8d](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)  | CPU, iGPU |
| [resnext50_32x4d](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)  | CPU, iGPU |
| [shufflenet_v2_x0_5](https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py)  | CPU, iGPU |
| [shufflenet_v2_x1_0](https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py)  | CPU, iGPU |
| [squeezenet1_0](https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py)  | CPU, iGPU |
| [squeezenet1_1](https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py)  | CPU, iGPU |
| [vgg11](https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py)  | CPU, iGPU |
| [vgg11_bn](https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py)  | CPU, iGPU |
| [vgg13](https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py)  | CPU, iGPU |
| [vgg13_bn](https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py)  | CPU, iGPU |
| [vgg16](https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py)  | CPU, iGPU |
| [vgg16_bn](https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py)  | CPU, iGPU |
| [vgg19](https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py)  | CPU, iGPU |
| [vgg19_bn](https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py)  | CPU, iGPU |
| [wide_resnet101_2](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)  | CPU, iGPU |
| [wide_resnet50_2](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)  | CPU, iGPU |

## HuggingFace models
| Model Name | Supported Devices |
|---|---|
| [bert-base-cased](https://huggingface.co/bert-base-cased)  | CPU, iGPU |
| [bert-base-chinese](https://huggingface.co/bert-base-chinese)  | CPU, iGPU |
| [bert-base-japanese-char](https://huggingface.co/cl-tohoku/bert-base-japanese-char)  | CPU, iGPU |
| [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased)  | CPU, iGPU |
| [bert-base-uncased](https://huggingface.co/bert-base-uncased)  | CPU, iGPU |
| [distilbert-base-cased](https://huggingface.co/distilbert-base-cased)  | CPU, iGPU |
| [distilbert-base-multilingual-cased](https://huggingface.co/distilbert-base-multilingual-cased)  | CPU, iGPU |
| [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)  | CPU, iGPU |
| [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)  | CPU, iGPU |
| [gpt2](https://huggingface.co/gpt2)  | CPU, iGPU |
| [roberta-base](https://huggingface.co/roberta-base)  | CPU, iGPU |
| [roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2)  | CPU, iGPU |
| [twitter-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)  | CPU, iGPU |



## Open Model Zoo
| Model Name | Supported Devices |
|---|---|
| [bert-base-ner](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/bert-base-ner)  | CPU, iGPU |
| [colorization-siggraph](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/colorization-siggraph)  | CPU, iGPU |
| [colorization-v2](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/colorization-v2)  | CPU, iGPU |
| [dla-34](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/dla-34)  | CPU, iGPU |
| [f3net](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/f3net)  | CPU, iGPU |
| [faceboxes-pytorch](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/faceboxes-pytorch)  | CPU, iGPU |
| [hbonet-0.25](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/hbonet-0.25)  | CPU, iGPU |
| [hbonet-0.5](https://docs.openvino.ai/2021.3/omz_models_model_hbonet_0_5.html)  | CPU, iGPU |
| [hbonet-1.0](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/hbonet-1.0)  | CPU, iGPU |
| [higher-hrnet-w32-human-pose-estimation](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/higher-hrnet-w32-human-pose-estimation)  | CPU, iGPU |
| [hrnet-v2-c1-segmentation](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/hrnet-v2-c1-segmentation) | iGPU |
| [human-pose-estimation-3d-0001](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/human-pose-estimation-3d-0001)  | CPU, iGPU |
| [midasnet](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/midasnet)  | CPU |
| [nfnet-f0](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/nfnet-f0)  | CPU, iGPU |
| [repvgg-a0](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/repvgg-a0)  | CPU, iGPU |
| [repvgg-b1](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/repvgg-b1)  | CPU |
| [repvgg-b3](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/repvgg-b3)  | CPU, iGPU |
| [resnest-50-pytorch](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/resnest-50-pytorch)  | CPU, iGPU |
| [retinaface-resnet50-pytorch](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/retinaface-resnet50-pytorch)  | CPU, iGPU |
| [rexnet-v1-x1.0](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/rexnet-v1-x1.0)  | CPU, iGPU |
| [single-human-pose-estimation-0001](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/single-human-pose-estimation-0001)  | CPU, iGPU |
| [yolact-resnet50-fpn-pytorch](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/yolact-resnet50-fpn-pytorch)  | CPU, iGPU |

## Other Models
| Model Name | Supported Devices |
|---|---|
| [cbam](https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py)  | CPU, iGPU |
| [crnn](https://github.com/meijieru/crnn.pytorch/blob/master/models/crnn.py)  | CPU, iGPU |
| [dlrm](https://github.com/facebookresearch/dlrm)  | CPU, iGPU |
| [dpn68](https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/dpn.py)  | CPU, iGPU |
| [fbnetc_100](https://github.com/rwightman/gen-efficientnet-pytorch/blob/master/geffnet/gen_efficientnet.py)  | CPU, iGPU |
| [inceptionresnetv2](https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/inceptionresnetv2.py)  | CPU, iGPU |
| [mobilenetv1_ssd](https://github.com/qfgaohao/pytorch-ssd/blob/master/vision/ssd/mobilenetv1_ssd.py)  | CPU, iGPU |
| [ncf](https://github.com/mlcommons/training/tree/master/recommendation/pytorch)  | CPU, iGPU |
| [seresnext-50](https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py)  | CPU, iGPU |
| [spnasnet_100](https://github.com/rwightman/gen-efficientnet-pytorch/blob/master/geffnet/gen_efficientnet.py)  | CPU, iGPU |
| [vnet](https://github.com/zyody/vnet.pytorch/blob/master/vnet.py)  | CPU, iGPU |
| [yolov2](https://github.com/yjh0410/yolov2-yolov3_PyTorch)  | CPU, iGPU |
| [yolov3](https://github.com/eriklindernoren/PyTorch-YOLOv3)  | CPU, iGPU |
| [yolov3-tiny](https://github.com/eriklindernoren/PyTorch-YOLOv3)  | CPU, iGPU |
