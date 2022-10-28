This document provides a list of all validated models that are supported by **OpenVINO™ integration with Torch-ORT**. The list of supported models and performance is continuously evolving as we are optimizing models and enabling more operators. The performance gain depends on various factors such as model architecture, Intel<sup>®</sup> Platform (e.g. Xeon<sup>®</sup> or Core<sup>TM</sup>), Backend device (e.g. CPU, GPU or VPU) etc. 


## Torchvision Models
| Model Name | Supported Devices |
|---|---|
| [alexnet](https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py)  | CPU, iGPU |
| [convnext_base](https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py)  | CPU, iGPU |
| [convnext_large](https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py)  | CPU, iGPU |
| [convnext_small](https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py)  | CPU, iGPU |
| [convnext_tiny](https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py)  | CPU, iGPU |
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
| [efficientnet_v2_l](https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py)  | CPU, iGPU |
| [efficientnet_v2_m](https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py)  | CPU, iGPU |
| [efficientnet_v2_s](https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py)  | CPU, iGPU |
| [fasterrcnn_mobilenet_v3_large_fpn](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py)  | CPU  |
| [fasterrcnn_resnet50_fpn](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py)  | CPU |
| [fasterrcnn_resnet50_fpn_v2](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py)  | CPU |
| [fcn_resnet101](https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py)  | CPU, iGPU |
| [fcn_resnet50](https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py)  | CPU, iGPU |
| [googlenet](https://github.com/pytorch/vision/blob/main/torchvision/models/googlenet.py)  | CPU, iGPU |
| [inception_v3](https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py)  | CPU, iGPU |
| [keypointrcnn_resnet50_fpn](https://github.com/scnuhealthy/Pytorch-Keypoint-Detection/blob/master/keypoint_rcnn.py)  | CPU |
| [lraspp_mobilenet_v3_large](https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/lraspp.py)  | CPU, iGPU |
| [maskrcnn_resnet50_fpn](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/mask_rcnn.py)  | CPU |
| [maskrcnn_resnet50_fpn_v2](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/mask_rcnn.py)  | CPU |
| [mnasnet0_5](https://github.com/pytorch/vision/blob/main/torchvision/models/mnasnet.py)  | CPU, iGPU |
| [mnasnet0_75](https://github.com/pytorch/vision/blob/main/torchvision/models/mnasnet.py)  | CPU, iGPU |
| [mnasnet1_0](https://github.com/pytorch/vision/blob/main/torchvision/models/mnasnet.py)  | CPU, iGPU |
| [mnasnet1_3](https://github.com/pytorch/vision/blob/main/torchvision/models/mnasnet.py)  | CPU, iGPU |
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
| [resnext101_64x4d](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py)  | CPU, iGPU |
| [resnext50_32x4d](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)  | CPU, iGPU |
| [shufflenet_v2_x0_5](https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py)  | CPU, iGPU |
| [shufflenet_v2_x1_0](https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py)  | CPU, iGPU |
| [shufflenet_v2_x1_5](https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py)  | CPU, iGPU |
| [shufflenet_v2_x2_0](https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py)  | CPU, iGPU |
| [squeezenet1_0](https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py)  | CPU, iGPU |
| [squeezenet1_1](https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py)  | CPU, iGPU |
| [ssd300_vgg16](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssd.py)  | CPU, iGPU |
| [ssdlite320_mobilenet_v3_large](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssdlite.py)  | CPU, iGPU |
| [swin_b](https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py)  | CPU, iGPU |
| [swin_s](https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py)  | CPU, iGPU |
| [swin_t](https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py)  | CPU, iGPU |
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
| [albert-base-v2](https://huggingface.co/albert-base-v2?text=The+goal+of+life+is+%5BMASK%5D.)  | CPU, iGPU |
| [allenai-scibert_scivocab_uncased](https://huggingface.co/allenai/scibert_scivocab_uncased)  | CPU, iGPU |
| [bert-base-cased](https://huggingface.co/bert-base-cased)  | CPU, iGPU |
| [bert-base-chinese](https://huggingface.co/bert-base-chinese)  | CPU, iGPU |
| [bert-base-japanese-char](https://huggingface.co/cl-tohoku/bert-base-japanese-char)  | CPU, iGPU |
| [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased)  | CPU, iGPU |
| [bert-base-multilingual-uncased](https://huggingface.co/bert-base-multilingual-uncased)  | CPU, iGPU |
| [bert-base-uncased](https://huggingface.co/bert-base-uncased)  | CPU, iGPU |
| [bert-large-uncased](https://huggingface.co/bert-large-uncased?text=Paris+is+the+%5BMASK%5D+of+France.)  | CPU, iGPU |
| [cl-tohoku-bert-base-japanese-whole-word-masking](https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking?text=%E6%9D%B1%E5%8C%97%E5%A4%A7%E5%AD%A6%E3%81%A7%5BMASK%5D%E3%81%AE%E7%A0%94%E7%A9%B6%E3%82%92%E3%81%97%E3%81%A6%E3%81%84%E3%81%BE%E3%81%99%E3%80%82)  | CPU, iGPU |
| [daigo-bert-base-japanese-sentiment](https://huggingface.co/daigo/bert-base-japanese-sentiment)  | CPU, iGPU |
| [DeepPavlov-rubert-base-cased-conversational](https://huggingface.co/DeepPavlov/rubert-base-cased-conversational)  | CPU, iGPU |
| [deepset-roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2?context=The+Amazon+rainforest+%28Portuguese%3A+Floresta+Amaz%C3%B4nica+or+Amaz%C3%B4nia%3B+Spanish%3A+Selva+Amaz%C3%B3nica%2C+Amazon%C3%ADa+or+usually+Amazonia%3B+French%3A+For%C3%AAt+amazonienne%3B+Dutch%3A+Amazoneregenwoud%29%2C+also+known+in+English+as+Amazonia+or+the+Amazon+Jungle%2C+is+a+moist+broadleaf+forest+that+covers+most+of+the+Amazon+basin+of+South+America.+This+basin+encompasses+7%2C000%2C000+square+kilometres+%282%2C700%2C000+sq+mi%29%2C+of+which+5%2C500%2C000+square+kilometres+%282%2C100%2C000+sq+mi%29+are+covered+by+the+rainforest.+This+region+includes+territory+belonging+to+nine+nations.+The+majority+of+the+forest+is+contained+within+Brazil%2C+with+60%25+of+the+rainforest%2C+followed+by+Peru+with+13%25%2C+Colombia+with+10%25%2C+and+with+minor+amounts+in+Venezuela%2C+Ecuador%2C+Bolivia%2C+Guyana%2C+Suriname+and+French+Guiana.+States+or+departments+in+four+nations+contain+%22Amazonas%22+in+their+names.+The+Amazon+represents+over+half+of+the+planet%27s+remaining+rainforests%2C+and+comprises+the+largest+and+most+biodiverse+tract+of+tropical+rainforest+in+the+world%2C+with+an+estimated+390+billion+individual+trees+divided+into+16%2C000+species.&question=Which+name+is+also+used+to+describe+the+Amazon+rainforest+in+English%3F)  | CPU, iGPU |
| [deepset-xlm-roberta-base-squad2](https://huggingface.co/deepset/xlm-roberta-base-squad2?context=The+Amazon+rainforest+%28Portuguese%3A+Floresta+Amaz%C3%B4nica+or+Amaz%C3%B4nia%3B+Spanish%3A+Selva+Amaz%C3%B3nica%2C+Amazon%C3%ADa+or+usually+Amazonia%3B+French%3A+For%C3%AAt+amazonienne%3B+Dutch%3A+Amazoneregenwoud%29%2C+also+known+in+English+as+Amazonia+or+the+Amazon+Jungle%2C+is+a+moist+broadleaf+forest+that+covers+most+of+the+Amazon+basin+of+South+America.+This+basin+encompasses+7%2C000%2C000+square+kilometres+%282%2C700%2C000+sq+mi%29%2C+of+which+5%2C500%2C000+square+kilometres+%282%2C100%2C000+sq+mi%29+are+covered+by+the+rainforest.+This+region+includes+territory+belonging+to+nine+nations.+The+majority+of+the+forest+is+contained+within+Brazil%2C+with+60%25+of+the+rainforest%2C+followed+by+Peru+with+13%25%2C+Colombia+with+10%25%2C+and+with+minor+amounts+in+Venezuela%2C+Ecuador%2C+Bolivia%2C+Guyana%2C+Suriname+and+French+Guiana.+States+or+departments+in+four+nations+contain+%22Amazonas%22+in+their+names.+The+Amazon+represents+over+half+of+the+planet%27s+remaining+rainforests%2C+and+comprises+the+largest+and+most+biodiverse+tract+of+tropical+rainforest+in+the+world%2C+with+an+estimated+390+billion+individual+trees+divided+into+16%2C000+species.&question=Which+name+is+also+used+to+describe+the+Amazon+rainforest+in+English%3F)  | CPU, iGPU |
| [distilbert-base-cased](https://huggingface.co/distilbert-base-cased)  | CPU, iGPU |
| [distilbert-base-cased-distilled-squad](https://huggingface.co/distilbert-base-cased-distilled-squad?context=My+name+is+Wolfgang+and+I+live+in+Berlin&question=Where+do+I+live%3F)  | CPU, iGPU |
| [distilbert-base-multilingual-cased](https://huggingface.co/distilbert-base-multilingual-cased)  | CPU, iGPU |
| [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)  | CPU, iGPU |
| [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)  | CPU, iGPU |
| [distilgpt2](https://huggingface.co/distilgpt2?text=My+name+is+Lewis+and+I+like+to)  | CPU |
| [distilroberta-base](https://huggingface.co/distilroberta-base?text=The+goal+of+life+is+%3Cmask%3E.)  | CPU, iGPU |
| [facebook-opt-125m](https://huggingface.co/facebook/opt-125m)  | CPU, iGPU |
| [gpt2](https://huggingface.co/gpt2)  | CPU |
| [Hate-speech-CNERG-indic-abusive-allInOne-MuRIL](https://huggingface.co/Hate-speech-CNERG/indic-abusive-allInOne-MuRIL?text=%E0%A6%AC%E0%A6%BE%E0%A6%99%E0%A6%BE%E0%A6%B2%E0%A6%BF%E0%A6%B0+%E0%A6%98%E0%A6%B0%E0%A7%87+%E0%A6%98%E0%A6%B0%E0%A7%87+%E0%A6%86%E0%A6%9C+%E0%A6%A8%E0%A6%AC%E0%A6%BE%E0%A6%A8%E0%A7%8D%E0%A6%A8+%E0%A6%89%E0%A7%8E%E0%A6%B8%E0%A6%AC%E0%A5%A4)  | CPU, iGPU |
| [Jean-Baptiste-camembert-ner](https://huggingface.co/Jean-Baptiste/camembert-ner?text=Je+m%27appelle+jean-baptiste+et+je+vis+%C3%A0+montr%C3%A9al)  | CPU, iGPU |
| [microsoft-codebert-base](https://huggingface.co/microsoft/codebert-base)  | CPU, iGPU |
| [microsoft-deberta-base](https://huggingface.co/microsoft/deberta-base?text=The+goal+of+life+is+%5BMASK%5D.)  | CPU |
| [mrm8488-t5-base-finetuned-question-generation-ap](https://huggingface.co/mrm8488/t5-base-finetuned-question-generation-ap?text=answer%3A+Manuel+context%3A+Manuel+has+created+RuPERTa-base+with+the+support+of+HF-Transformers+and+Google)  | CPU, iGPU |
| [nlptown-bert-base-multilingual-uncased-sentiment](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment?text=I+like+you.+I+love+you)  | CPU, iGPU |
| [prajjwal1-bert-tiny](https://huggingface.co/prajjwal1/bert-tiny)  | CPU, iGPU |
| [roberta-base](https://huggingface.co/roberta-base)  | CPU, iGPU |
| [roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2)  | CPU, iGPU |
| [sentence-transformers-all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)  | CPU, iGPU |
| [sentence-transformers-all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)  | CPU, iGPU |
| [sentence-transformers-bert-base-nli-mean-tokens](https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens)  | CPU, iGPU |
| [sentence-transformers-paraphrase-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2)  | CPU, iGPU |
| [t5-base](https://huggingface.co/t5-base?text=My+name+is+Sarah+and+I+live+in+London)  | iGPU |
| [t5-small](https://huggingface.co/t5-small?text=My+name+is+Sarah+and+I+live+in+London)  | CPU, iGPU |
| [tals-albert-xlarge-vitaminc-mnli](https://huggingface.co/tals/albert-xlarge-vitaminc-mnli?text=I+like+you.+I+love+you)  | CPU |
| [twitter-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)  | CPU, iGPU |
| [unitary-toxic-bert](https://huggingface.co/unitary/toxic-bert?text=I+like+you.+I+love+you)  | CPU, iGPU |
| [valhalla-t5-small-qa-qg-hl](https://huggingface.co/valhalla/t5-small-qa-qg-hl?text=generate+question%3A+%3Chl%3E+42+%3Chl%3E+is+the+answer+to+life%2C+the+universe+and+everything.+%3C%2Fs%3E)  | CPU, iGPU |
| [xlm-roberta-base](https://huggingface.co/xlm-roberta-base)  | iGPU |

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
| [repvgg-b3](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/repvgg-b3)  | CPU |
| [resnest-50-pytorch](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/resnest-50-pytorch)  | CPU, iGPU |
| [retinaface-resnet50-pytorch](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/retinaface-resnet50-pytorch)  | CPU, iGPU |
| [rexnet-v1-x1.0](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/rexnet-v1-x1.0)  | CPU, iGPU |
| [single-human-pose-estimation-0001](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/single-human-pose-estimation-0001)  | CPU, iGPU |
| [yolact-resnet50-fpn-pytorch](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/yolact-resnet50-fpn-pytorch)  | CPU, iGPU |

## Other Models
| Model Name | Supported Devices |
|---|---|
| [bert-large-cased](https://huggingface.co/bert-large-cased?text=Paris+is+the+%5BMASK%5D+of+France.)  | CPU, iGPU |
| [bert-base-cased+masked-lm](https://huggingface.co/bert-base-cased?text=Paris+is+the+%5BMASK%5D+of+France.)  | CPU, iGPU |
| [CaiT](https://github.com/eriklindernoren/PyTorch-YOLOv3)  | CPU, iGPU |
| [cbam](https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py)  | CPU, iGPU |
| [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)  | CPU, iGPU |
| [crnn](https://github.com/meijieru/crnn.pytorch/blob/master/models/crnn.py)  | CPU |
| [DeepViT](https://github.com/lucidrains/vit-pytorch#deep-vit)  | CPU, iGPU |
| [distilbert-base-cased+masked-lm](https://github.com/huggingface/transformers.git)  | CPU, iGPU |
| [distilbert-base-cased+multiple-choice](https://github.com/huggingface/transformers.git)  | CPU, iGPU |
| [dlrm](https://github.com/facebookresearch/dlrm)  | CPU |
| [dpn68](https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/dpn.py)  | CPU, iGPU |
| [facebook-bart-base](https://huggingface.co/facebook/bart-base)  | CPU, iGPU |
| [facebook-bart-large+question-answering](https://github.com/huggingface/transformers.git)  | CPU, iGPU |
| [fbnetc_100](https://github.com/rwightman/gen-efficientnet-pytorch/blob/master/geffnet/gen_efficientnet.py)  | CPU, iGPU |
| [google-electra-base-discriminator+multiple-choice](https://github.com/huggingface/transformers.git)  | CPU, iGPU |
| [google-electra-base-generator+sequence-classification](https://github.com/huggingface/transformers.git)  | CPU, iGPU |
| [google-electra-base-generator+token-classification](https://github.com/huggingface/transformers.git)  | CPU, iGPU |
| [gpt2+token-classification](https://github.com/huggingface/transformers.git)  | CPU |
| [inceptionresnetv2](https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/inceptionresnetv2.py)  | CPU |
| [LeViT](https://github.com/lucidrains/vit-pytorch#levit)  | CPU, iGPU |
| [mobilenetv1_ssd](https://github.com/qfgaohao/pytorch-ssd/blob/master/vision/ssd/mobilenetv1_ssd.py)  | CPU, iGPU |
| [nasnetalarge](https://github.com/Cadene/pretrained-models.pytorch.git)  | CPU, iGPU |
| [ncf](https://github.com/mlcommons/training/tree/master/recommendation/pytorch)  | CPU, iGPU |
| [polynet](https://github.com/Cadene/pretrained-models.pytorch.git)  | CPU, iGPU |
| [prajjwal1-bert-mini](https://huggingface.co/prajjwal1/bert-mini)  | CPU, iGPU |
| [reformer](https://github.com/lucidrains/reformer-pytorch.git)  | CPU |
| [se_resnet50](https://github.com/Cadene/pretrained-models.pytorch.git)  | CPU, iGPU |
| [senet154](https://github.com/Cadene/pretrained-models.pytorch.git)  | CPU, iGPU |
| [seresnext-50](https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py)  | CPU, iGPU |
| [spnasnet_100](https://github.com/rwightman/gen-efficientnet-pytorch/blob/master/geffnet/gen_efficientnet.py)  | CPU, iGPU |
| [ssd300](https://github.com/mlcommons/training.git)  | CPU, iGPU |
| [ssd-vgg16](https://github.com/qfgaohao/pytorch-ssd)  | CPU, iGPU |
| [T2TViT](https://github.com/lucidrains/vit-pytorch#token-to-token-vit)  | CPU, iGPU |
| [transformer-lt](https://github.com/pytorch/fairseq)  | CPU, iGPU |
| [unet](https://github.com/usuyama/pytorch-unet.git)  | CPU, iGPU |
| [vggm](https://github.com/Cadene/pretrained-models.pytorch.git)  | iGPU |
| [vnet](https://github.com/zyody/vnet.pytorch/blob/master/vnet.py)  | CPU |
| [yolov2](https://github.com/yjh0410/yolov2-yolov3_PyTorch)  | CPU, iGPU |
| [yolov3](https://github.com/eriklindernoren/PyTorch-YOLOv3)  | CPU, iGPU |
| [yolov3-tiny](https://github.com/eriklindernoren/PyTorch-YOLOv3)  | CPU, iGPU |

## Quantization Support (Experimental)
OpenVINO™ integration with Torch-ORT now supports INT8 models quantized using Post-Training Quantization (PTQ) through OpenVINO™ Neural Network Compression Framework (NNCF). This support is currently in an experimental state and performance optimizations are in progress.

Some examples of NNCF usage to produce quantized models can be found here: <https://github.com/openvinotoolkit/nncf/tree/develop/examples/experimental/torch>.
