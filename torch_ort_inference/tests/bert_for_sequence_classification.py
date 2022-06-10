# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import os
import numpy as np
import time
import pandas as pd

from transformers import BertTokenizer, AutoConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, BertConfig

import torch
from torch_ort import ORTInferenceModule, OpenVINOProviderOptions

ov_backend_precisions = {
    "CPU": ["FP32"],
    "GPU": ["FP32", "FP16"],
    #"MYRIAD": ["FP16"]
}

def get_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    return tokenizer

def preprocess_input(sentences):
    # Tokenization & Input Formatting
    # Load the BERT tokenizer.
    tokenizer = get_tokenizer()

    # Set the max length of encoded sentence.
    # 64 is slightly larger than the maximum training sentence length of 47...
    MAX_LEN = 512

    # Tokenize all of the sentences and map the tokens to their word IDs.
    input_ids = []
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    )

        # Pad our input tokens with value 0.
        if len(encoded_sent) < MAX_LEN:
            encoded_sent.extend([0]*(MAX_LEN-len(encoded_sent)))

        # Truncate to MAX_LEN
        if len(encoded_sent) > MAX_LEN:
            encoded_sent = encoded_sent[:MAX_LEN]

        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)

    input_ids = np.array(input_ids, dtype=np.longlong)

    # Create attention masks
    attention_masks = []

    # For each sentence...
    for sent in input_ids:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)
    return input_ids, attention_masks

def predict(model,prediction_dataloader):
    total_prediction_time=0
    results={}
    tokenizer = get_tokenizer()
    warm_up = True

    for batch in prediction_dataloader:    
        batch = tuple(t for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask = batch

        # Run inference
        with torch.no_grad():
            #warm-up
            if warm_up:
                model(b_input_ids, b_input_mask)
                warm_up=False
            #infer
            t0 = time.time()
            outputs =  model(b_input_ids, b_input_mask)
            t1 = time.time() - t0
            total_prediction_time += t1

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs.logits
        
        # Move logits
        logits = logits.detach().cpu().numpy()
        # predictions
        pred_flat = np.argmax(logits, axis=1).flatten()
        for i in range(len(b_input_ids)):
            orig_sent = tokenizer.decode(b_input_ids[i], skip_special_tokens=True)
            results[orig_sent] = pred_flat[i]
        
    print("\n Top Results: \n")
    count=0
    for k, v in results.items():
        print("\t{!r} : {!r}".format(k,v))
        if count == 20: break
        count = count + 1
    print("\nPrediction took: {:.4f}s".format(total_prediction_time))

def load_pred_dataset(args):

    # Input can be a single sentence, list of single sentences in a .tsv file.
    if args.input is not None:
        sentences = [args.input]
    elif args.input_file is not None:
        if not os.path.exists(args.input_file):
                raise ValueError('Invalid model path: %s' % args.input_file)
        df = pd.read_csv(args.input_file, delimiter='\t', header=None, names=['Id', 'Sentence'], skiprows=1)
        sentences = df.Sentence.values
    else:
        print("Input not provided! Using default input")
        sentences = ["This is a sample input."]

    input_ids,attention_masks = preprocess_input(sentences)

    input_ids = torch.tensor(input_ids)
    input_mask = torch.tensor(attention_masks)

    #Create DataLoader for prediction dataset
    prediction_data = TensorDataset(input_ids, input_mask)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=args.batch_size)

    return prediction_dataloader

def main():
    # 1. Basic setup
    parser = argparse.ArgumentParser(description='PyTorch BERT Sequence Classification Example')
    parser.add_argument('--pytorch-only', action='store_true', default=False,
                        help='disables ONNX Runtime inference')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for inference (default: 1)')
    parser.add_argument('--export-onnx-graphs', action='store_true', default=False,
                        help='export ONNX graphs to current directory')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to fine tuned model for prediction')
    parser.add_argument('--input', type=str, default=None,
                        help="Input sentence for prediction")
    parser.add_argument('--input-file', type=str, default=None,
                        help="Input file in .tsv format for prediction")
    parser.add_argument('--provider', type=str,
                        help="ONNX Runtime Execution Provider for inference")
    parser.add_argument('--backend', type=str, 
                        help="OpenVINO target device for Inference.")
    parser.add_argument('--precision', type=str,
                        help="OpenVINO target device precision for prediction")

    args = parser.parse_args()

    # parameters validation
    if args.provider != "openvino": 
        raise Exception("Invalid Provider! Set openvino as provider")
    else:
        if (args.backend is not None) and (args.backend not in list(ov_backend_precisions.keys())):
            raise Exception("Invalid backend string. Valid values are:", list(ov_backend_precisions.keys())) 
    
        if (args.precision is not None) and (args.precision not in ov_backend_precisions[args.backend]):
            raise Exception("Invalid precision for provided backend")

        if not (args.backend and args.precision):
            print("Please provide both backend and precision. If not default values are taken as CPU and FP32")
            args.backend = "CPU"
            args.precision = "FP32"

    # 2. Dataloader
    # Load input dataset/file for prediction
    prediction_dataloader  = load_pred_dataset(args)

    # 3. Load Model
    # If model path is not provided, use the model
    # Else use a pretrained model fine-tuned on CoLA dataset from huggingface model hub
    if args.model_path is None:
        model = BertForSequenceClassification.from_pretrained('textattack/bert-base-uncased-CoLA')       
    else:
        if not os.path.exists(args.model_path):
            raise ValueError('Invalid model path: %s' % args.model_path)
        
        # Load BertForSequenceClassification, the pretrained BERT model with a single
        # linear classification layer on top.
        config = AutoConfig.from_pretrained(
                "bert-base-uncased",
                num_labels=2,
                output_attentions = False, # Whether the model returns attentions weights.
                output_hidden_states = False, # Whether the model returns all hidden-states.
        )
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
            config=config
        )
        model.load_state_dict(torch.load(args.model_path))

    if not args.pytorch_only:
        provider_options = OpenVINOProviderOptions(provider=args.provider, backend=args.backend, precision=args.precision)
        model = ORTInferenceModule(model, provider_options=provider_options)

    # 4. Predict
    predict(model, prediction_dataloader)

if __name__ == '__main__':
    main()