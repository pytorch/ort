import os
import sys
import time
import torch
import wget
import argparse
from PIL import Image
from torchvision import transforms
import torchvision.models as models
from torch_ort import ORTInferenceModule, OpenVINOProviderOptions

ov_backend_precisions = {
    "CPU": ["FP32"],
    "GPU": ["FP32", "FP16"],
    "MYRIAD": ["FP16"]
}

def download_labels(labels):
    if not labels:
        labelsUrl = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
        labels = "imagenet_classes.txt"
        if not os.path.exists(labels):
            # Download the file (if we haven't already)
            wget.download(labelsUrl)
        else:
            print("\nReusing downloaded imagenet labels")

    # Read the categories
    with open(labels, "r") as f:
        categories = [s.strip() for s in f.readlines()]
        return categories

def preprocess(input):
    transform = transforms.Compose([
     transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
    )])
    return transform(input)

def predict(model,image,categories):
    # warmup
    for _ in range(5):
        outputs = model(image)

    # Start inference
    t0 = time.time()
    outputs = model(image)
    t1 = time.time() - t0
    print("\nInference time: {:.4f}ms\n".format(t1 * 1000))

    # The output has unnormalized scores. Run a softmax on it for probabilities.
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    print("Labels , Probabilities:")
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]],top5_prob[i].item())

def main():
    # 1. Basic setup
    parser = argparse.ArgumentParser(description='PyTorch Image Classification Example')
    parser.add_argument('--pytorch-only', action='store_true', default=False,
                        help='disables ONNX Runtime inference')
    parser.add_argument('--labels', type=str, default=None,
                        help="labels file")
    parser.add_argument('--input', type=str, required = True,
                        help="input image for inference")
    parser.add_argument('--provider', type=str,
                        help="ONNX Runtime Execution Provider for inference")
    parser.add_argument('--backend', type=str,
                        help="Backend for inference")
    parser.add_argument('--precision', type=str,
                        help="Precision for prediction")
    args = parser.parse_args()

    # parameters validation
    if args.provider is None: 
        print("Using default execution provider CPU, to use OpenVINOExecutionProvider set provider as openvino")
    
    if (args.backend is not None) and (args.backend not in list(ov_backend_precisions.keys())):
        raise Exception("Invalid backend string. Valid values are:", list(ov_backend_precisions.keys())) 
    
    if (args.precision is not None) and (args.precision not in ov_backend_precisions[args.backend]):
        raise Exception("Invalid precision for provided backend")

    if not (args.backend and args.precision):
        print("Please provide both backend and precision. If not default values are taken as CPU and FP32")


    # 2. Download and load the model
    model = models.resnet50(pretrained=True)
    if not args.pytorch_only:
        provider_options = OpenVINOProviderOptions(provider=args.provider, backend=args.backend, precision=args.precision)
        model = ORTInferenceModule(model, provider_options=provider_options)

    # Convert model for evaluation
    model.eval()

    # 3. Download ImageNet labels
    categories = download_labels(args.labels)
    
    # 4. Read input image and preprocess
    if not args.input:
        raise ValueError("Input image not provided!")
    img = Image.open(args.input)
    img_trans = preprocess(img)
    #Adding batch dimension (size 1)
    img_trans = torch.unsqueeze(img_trans, 0)

    # 5. Predict
    predict(model,img_trans,categories)

if __name__ == '__main__':
    main()