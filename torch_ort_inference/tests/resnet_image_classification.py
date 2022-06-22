# -------------------------------------------------------------------------
# Copyright (C) 2022 Intel Corporation
# Licensed under the MIT License
# --------------------------------------------------------------------------

import os
import time
import torch
import wget
import argparse
from PIL import Image
from torchvision import transforms
import torchvision.models as models
from torch_ort import ORTInferenceModule, OpenVINOProviderOptions

ov_backend_precisions = {"CPU": ["FP32"], "GPU": ["FP32", "FP16"], "MYRIAD": ["FP16"]}
inference_execution_providers = ["openvino"]

def download_labels(labels):
    if not labels:
        labels = "imagenet_classes.txt"
        if not os.path.exists(labels):
            labelsUrl = (
                "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
            )
            # Download the file (if we haven't already)
            wget.download(labelsUrl)
        else:
            print("\nReusing downloaded imagenet labels")

    # Read the categories
    with open(labels, "r") as f:
        categories = [s.strip() for s in f.readlines()]
        return categories


def preprocess(img):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(img)


def infer(model, image, categories):
    # warmup
    model(image)

    # Start inference
    t0 = time.time()
    outputs = model(image)
    t1 = time.time() - t0
    print("\nInference time: {:.4f}ms\n".format(t1 * 1000))

    # The output has unnormalized scores. Run a softmax on it for probabilities.
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    print("Top 5 Results: \nLabels , Probabilities:")
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())


def main():
    # 1. Basic setup
    parser = argparse.ArgumentParser(description="PyTorch Image Classification Example")

    parser.add_argument(
        "--pytorch-only",
        action="store_true",
        default=False,
        help="disables ONNX Runtime inference",
    )
    parser.add_argument(
        "--labels",
        type=str,
        help="path to labels file")
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="path to input image file"
    )
    parser.add_argument(
        "--provider",
        type=str,
        help="ONNX Runtime Execution Provider",
    )
    parser.add_argument(
        "--backend",
        type=str,
        help="OpenVINO target device (CPU, GPU or MYRIAD)"
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="OpenVINO target device precision (FP16 or FP32)"
    )

    args = parser.parse_args()

    # parameters validation
    if not args.pytorch_only:
        if args.provider is None or args.provider == "openvino":
            if args.backend and args.precision:
                if args.backend not in list(ov_backend_precisions.keys()):
                    raise Exception(
                        "Invalid backend. Valid values are: {}".format(
                            list(ov_backend_precisions.keys())))
                if args.precision not in ov_backend_precisions[args.backend]:
                    raise Exception("Invalid precision for provided backend. Valid values are: {}".format(
                    list(ov_backend_precisions[args.backend])))
            elif args.backend or args.precision:
                raise Exception(
                    "Please specify both backend and precision to override default options.\n"
                )
            else:
                print("OpenVINOExecutionProvider is enabled with CPU and FP32 by default.")
        else:
            raise Exception("Invalid execution provider!! Available providers are: {}".format(inference_execution_providers))
    else:
        print("ONNXRuntime inference is disabled.")
        if args.provider or args.precision or args.backend:
            raise Exception("provider, backend, precision arguments are not applicable for --pytorch-only option")

    # 2. Read input image file and preprocess
    if not args.input_file:
        raise ValueError("Path to input image not provided!")
    if not os.path.exists(args.input_file):
        raise ValueError("Invalid input file path")
    img = Image.open(args.input_file)
    img_trans = preprocess(img)
    # Adding batch dimension (size 1)
    img_trans = torch.unsqueeze(img_trans, 0)

    # 3. Download and load the model
    model = models.resnet50(pretrained=True)
    if not args.pytorch_only:
        if (args.provider == "openvino" or args.provider is None) \
            and (args.backend and args.precision):
            provider_options = OpenVINOProviderOptions(
                backend=args.backend, precision=args.precision
            )
            model = ORTInferenceModule(model, provider_options=provider_options)
        else:
            model = ORTInferenceModule(model)

    # Convert model for evaluation
    model.eval()

    # 4. Download ImageNet labels
    categories = download_labels(args.labels)

    # 5. Infer
    infer(model, img_trans, categories)
    img.close()

if __name__ == "__main__":
    main()
