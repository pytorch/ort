This document provides a list of all validated models that are supported by **OpenVINO™ integration with Torch-ORT**. The list of supported models and performance is continuously evolving as we are optimizing models and enabling more operators. The performance gain depends on various factors such as model architecture, Intel<sup>®</sup> Platform (e.g. Xeon<sup>®</sup> or Core<sup>TM</sup>), Backend device (e.g. CPU, GPU or VPU) etc. 


## Torcvision Models
| Model Name | Supported Devices |
|---|---|
| [shufflenet_v2_x1_0]()  | CPU, iGPU |
| [shufflenet_v2_x0_5]()  | CPU, iGPU |
| [wide_resnet101_2]()  | CPU, iGPU |
| [resnext50_32x4d]()  | CPU, iGPU |
| [deeplabv3_mobilenet_v3_large]()  | CPU, iGPU |
| [vgg19]()  | CPU, iGPU |
| [densenet161]()  | CPU, iGPU |
| [regnet_y_800mf]()  | CPU, iGPU |
| [resnext101_32x8d]()  | CPU, iGPU |
| [regnet_y_16gf]()  | CPU, iGPU |
| [alexnet]()  | CPU, iGPU |
| [resnet_3d]()  | CPU, iGPU |
| [efficientnet_b3]()  | CPU, iGPU |
| [resnet101]()  | CPU, iGPU |
| [mobilenet_v2]()  | CPU, iGPU |
| [regnet_y_400mf]()  | CPU, iGPU |
| [resnet18]()  | CPU, iGPU |
| [vgg16]()  | CPU, iGPU |
| [resnet50]()  | CPU, iGPU |
| [mobilenet_v3_large]()  | CPU, iGPU |
| [regnet_x_3_2gf]()  | CPU, iGPU |
| [deeplabv3_resnet101]()  | CPU, iGPU |
| [deeplabv3_resnet50]()  | CPU, iGPU |
| [fcn_resnet101]()  | CPU, iGPU |
| [fcn_resnet50]()  | CPU, iGPU |
| [googlenet]()  | CPU, iGPU |
| [lraspp_mobilenet_v3_large]()  | CPU, iGPU |
| [vgg13_bn]()  | CPU, iGPU |
| [regnet_y_8gf]()  | CPU, iGPU |
| [vgg19_bn]()  | CPU, iGPU |
| [mnasnet0_5]()  | CPU, iGPU |
| [regnet_x_8gf]()  | CPU, iGPU |
| [mnasnet1_0]()  | CPU, iGPU |
| [inception_v3]()  | CPU, iGPU |
| [squeezenet1_1]()  | CPU, iGPU |
| [regnet_y_3_2gf]()  | CPU, iGPU |
| [regnet_x_800mf]()  | CPU, iGPU |
| [vgg13]()  | CPU, iGPU |
| [densenet169]()  | CPU, iGPU |
| [resnet34]()  | CPU, iGPU |
| [resnet152]()  | CPU, iGPU |
| [resnet_2plus1d]()  | CPU, iGPU |
| [regnet_x_400mf]()  | CPU, iGPU |
| [densenet121]()  | CPU, iGPU |
| [vgg16_bn]()  | CPU, iGPU |
| [efficientnet_b0]()  | CPU, iGPU |
| [vgg11]()  | CPU, iGPU |
| [densenet201]()  | CPU, iGPU |
| [vgg11_bn]()  | CPU, iGPU |
| [regnet_x_1_6gf]()  | CPU, iGPU |
| [mobilenet_v3_small]()  | CPU, iGPU |
| [regnet_y_1_6gf]()  | CPU, iGPU |
| [regnet_x_16gf]()  | CPU, iGPU |
| [efficientnet_b7]()  | CPU, iGPU |
| [regnet_x_32gf]()  | CPU, iGPU |
| [efficientnet_b1]()  | CPU, iGPU |
| [efficientnet_b2]()  | CPU, iGPU |
| [wide_resnet50_2]()  | CPU, iGPU |
| [efficientnet_b4]()  | CPU, iGPU |
| [efficientnet_b5]()  | CPU, iGPU |
| [squeezenet1_0]()  | CPU, iGPU |
| [efficientnet_b6]()  | CPU, iGPU |
| [resnet_mixed_conv_3d]()  | CPU, iGPU |

## HuggingFace models
| Model Name | Supported Devices |
|---|---|
| [distilbert-base-multilingual-cased]()  | CPU, iGPU |
| [bert-base-multilingual-cased]()  | CPU, iGPU |
| [bert-base-cased]()  | CPU, iGPU |
| [roberta-base]()  | CPU, iGPU |
| [bert-base-chinese]()  | CPU, iGPU |
| [twitter-roberta-base-sentiment]()  | CPU, iGPU |
| [gpt2]()  | CPU, iGPU |
| [bert-base-uncased]()  | CPU, iGPU |
| [distilbert-base-uncased]()  | CPU, iGPU |
| [roberta-base-squad2]()  | CPU, iGPU |
| [distilbert-base-cased]()  | CPU, iGPU |
| [bert-base-japanese-char]()  | CPU, iGPU |
| [distilbert-base-uncased-finetuned-sst-2-english]()  | CPU, iGPU |

## Open Model Zoo
| Model Name | Supported Devices |
|---|---|
| [retinaface-resnet50-pytorch]()  | CPU, iGPU |
| [single-human-pose-estimation-0001]()  | CPU, iGPU |
| [repvgg-a0]()  | CPU, iGPU |
| [hbonet-0.25]()  | CPU, iGPU |
| [faceboxes-pytorch]()  | CPU, iGPU |
| [hbonet-0.5]()  | CPU, iGPU |
| [colorization-siggraph]()  | CPU, iGPU |
| [bert-base-ner]()  | CPU, iGPU |
| [nfnet-f0]()  | CPU, iGPU |
| [hrnet-v2-c1-segmentation]()  | iGPU |
| [higher-hrnet-w32-human-pose-estimation]()  | CPU, iGPU |
| [repvgg-b3]()  | CPU, iGPU |
| [human-pose-estimation-3d-0001]()  | CPU, iGPU |
| [rexnet-v1-x1.0]()  | CPU, iGPU |
| [midasnet]()  | CPU |
| [dla-34]()  | CPU, iGPU |
| [colorization-v2]()  | CPU, iGPU |
| [yolact-resnet50-fpn-pytorch]()  | CPU, iGPU |
| [resnest-50-pytorch]()  | CPU, iGPU |
| [hbonet-1.0]()  | CPU, iGPU |
| [repvgg-b1]()  | CPU, iGPU |
| [f3net]()  | CPU, iGPU |