# -------------------------------------------------------------------------
# Copyright (C) 2022 Intel Corporation
# Licensed under the MIT License
# --------------------------------------------------------------------------

import argparse
import os
import numpy as np
import time
import pandas as pd
import pathlib

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

import torch
from torch_ort import ORTInferenceModule, OpenVINOProviderOptions

ov_backend_precisions = {"CPU": ["FP32"], "GPU": ["FP32", "FP16"]}
inference_execution_providers = ["openvino"]

def preprocess_input(tokenizer, sentences):
    # Tokenization & Input Formatting
    # Config: "do_lower_case": true, "model_max_length": 512
    inputs = []

    MAX_LEN = 64

    for sentence in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
                            sentence,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    )

        # Pad our input tokens with value 0.
        if len(encoded_sent) < MAX_LEN:
            encoded_sent.extend([0]*(MAX_LEN-len(encoded_sent)))

        # Truncate to MAX_LEN
        if len(encoded_sent) > MAX_LEN:
            print("WARNING: During preprocessing, number of tokens for the sentence {}"\
                "exceedeed MAX LENGTH {}. This might impact accuracy of the results".format(
                sentence,
                MAX_LEN
            ))
            encoded_sent = encoded_sent[:MAX_LEN]

        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in encoded_sent]

        # Store the input ids and attention masks for the sentence.
        inputs.append({'input_ids': torch.unsqueeze(torch.tensor(encoded_sent),0),
                'attention_mask': torch.unsqueeze(torch.tensor(att_mask),0)})

    return inputs


def infer(model, sentences, inputs):
    num_sentences = len(sentences)
    total_infer_time = 0
    results = {}

    # Run inference
    for i in range(num_sentences):
        input_ids = (inputs[i])['input_ids']
        attention_masks = (inputs[i])['attention_mask']
        with torch.no_grad():
            # warm-up
            if i == 0:
               t0 = time.time()
               model(input_ids, attention_masks)
            # infer
            t0 = time.time()
            outputs = model(input_ids, attention_masks)
            t1 = time.time() - t0
            total_infer_time += t1

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs.logits

        # Move logits
        logits = logits.detach().cpu().numpy()

        # predictions
        pred_flat = np.argmax(logits, axis=1).flatten()
        orig_sent = sentences[i]
        results[orig_sent] = pred_flat[0]

    print("\n Number of sentences: {}".format(num_sentences))
    if num_sentences > 20:
        print(" First 20 results:")
    print("\t Grammar correctness label (0=unacceptable, 1=acceptable)\n")
    count = 0
    for k, v in results.items():
        print("\t{!r} : {!r}".format(k, v))
        if count == 20:
            break
        count = count + 1
    print("\n Average inference time: {:.4f}ms".format((total_infer_time/num_sentences)*1000))
    print(" Total Inference time: {:.4f}ms".format(total_infer_time * 1000))

def main():
    # 1. Basic setup
    parser = argparse.ArgumentParser(
        description="PyTorch BERT Sequence Classification Example"
    )
    parser.add_argument(
        "--pytorch-only",
        action="store_true",
        default=False,
        help="disables ONNX Runtime inference",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="input sentence")

    parser.add_argument(
        "--input-file",
        type=str,
        help="path to input file in .tsv format",
    )
    parser.add_argument(
        "--provider",
        type=str,
        help="ONNX Runtime Execution Provider",
    )
    parser.add_argument(
        "--backend",
        type=str,
        help="OpenVINO target device (CPU, GPU)."
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="OpenVINO target device precision (FP16 or FP32)"
    )

    args = parser.parse_args()

    # parameters validation
    if not args.pytorch_only:
        if args.provider is None:
            print("OpenVINOExecutionProvider is enabled with CPU and FP32 by default.")
            if args.backend or args.precision:
                raise ValueError("Provider not specified!! Please specify provider arg along with backend and precision.")
        elif args.provider == "openvino":
            if args.backend and args.precision:
                if args.backend not in list(ov_backend_precisions.keys()):
                    raise ValueError(
                        "Invalid backend. Valid values are: {}".format(
                            list(ov_backend_precisions.keys())))
                if args.precision not in ov_backend_precisions[args.backend]:
                    raise ValueError("Invalid precision for provided backend. Valid values are: {}".format(
                    list(ov_backend_precisions[args.backend])))
            elif args.backend or args.precision:
                raise ValueError(
                    "Please specify both backend and precision to override default options.\n"
                )
            else:
                print("OpenVINOExecutionProvider is enabled with CPU and FP32 by default.")
        else:
            raise ValueError("Invalid execution provider!! Available providers are: {}".format(inference_execution_providers))
    else:
        print("ONNXRuntime inference is disabled.")
        if args.provider or args.precision or args.backend:
            raise ValueError("provider, backend, precision arguments are not applicable for --pytorch-only option.")

    # 2. Read input sentence(s)
    # Input can be a single sentence, list of single sentences in a .tsv file.
    if args.input and args.input_file:
        raise ValueError("Please provide either input or input file for inference.")

    if args.input is not None:
        sentences = [args.input]
    elif args.input_file is not None:
        file_name = args.input_file
        if not os.path.exists(file_name):
            raise ValueError("Invalid input file path: %s" % file_name)
        if os.stat(file_name).st_size == 0:
            raise ValueError("Input file is empty!!")
        name, ext = os.path.splitext(file_name)
        if ext != ".tsv":
            raise ValueError("Invalid input file format. Please provide .tsv file.")
        df = pd.read_csv(
            file_name,
            delimiter="\t",
            header=None,
            names=["Id", "Sentence"],
            skiprows=1,
        )
        sentences = df.Sentence.values
    else:
        print("Input not provided! Using default input...")
        sentences = ["This is a BERT sample.","User input is valid not."]

    # 3. Load Model
    # Pretrained model fine-tuned on CoLA dataset from huggingface model hub to predict grammar correctness
    model = AutoModelForSequenceClassification.from_pretrained(
        "textattack/bert-base-uncased-CoLA"
    )

    if not args.pytorch_only:
        if args.provider == "openvino" and (args.backend and args.precision):
            provider_options = OpenVINOProviderOptions(
                backend=args.backend, precision=args.precision
            )
            model = ORTInferenceModule(model, provider_options=provider_options)
        else:
            model = ORTInferenceModule(model)

    # Convert model for evaluation
    model.eval()

    # 4. Load Tokenizer & Preprocess input sentences
    tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-CoLA")
    inputs = preprocess_input(tokenizer, sentences)

    # 5. Infer
    infer(model, sentences, inputs)


if __name__ == "__main__":
    main()
