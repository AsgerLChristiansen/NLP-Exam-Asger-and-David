# Run the setup script first!
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse
import numpy as np
import torch
import datasets
from datasets.dataset_dict import DatasetDict

from gensim.models.keyedvectors import KeyedVectors
from torch import nn
import random
from src.data import arg_inputs
from src.train import train_model
random.seed(1337)

# Import training script (and allow for changes to it)
import importlib

#Import model and optimizer
from transformers import AutoModelForSequenceClassification

from torch.optim import AdamW
from datasets import load_from_disk

from torch.utils.data import DataLoader

def main(old_model_name:str = None, new_model_name:str = "My model", dataset:str = "docs_train.pt"):
    
    train = load_from_disk("src/preprocessed/" + dataset)
    # Get length of input
    len_train = len(train["input_ids"])
    #len_val = len(val["input_ids"])
    if old_model_name:
        model = torch.load('src/models/' + old_model_name)
    else:
        # Let's train a model!
        bert_model = "bert-base-uncased"
        model = AutoModelForSequenceClassification.from_pretrained(bert_model, num_labels=11) # We have 11 labels.
    optimizer = AdamW(model.parameters(), lr=5e-5)
    # Training loop (Batch size of 32. Data is subsetted into chunks of 320 to avoid crashing issues)   
    # Runs only one epoch due to technical limitations. LEss likely to crash this way.
    i = 0
    j = 0
    while True:
        if i + 320 < len_train:
            small_train = train.select(range(i, i +320))
            train_dataloader = torch.utils.data.DataLoader(small_train, batch_size=32)
            train_model(model, train_dataloader, optimizer, new_model_name)
        else:
            small_train = train.select(range(i, len_train))
            train_dataloader = torch.utils.data.DataLoader(small_train, batch_size=32)
            train_model(model, train_dataloader, optimizer, new_model_name)
            break
        i += 320
        j += 1
        if j % 10 == 0:
            print("100 batches processed!")

# Command line functionality
if __name__ == "__main__":
    arguments = arg_inputs()
    main(old_model_name = arguments.old_model_name, new_model_name = arguments.new_model_name, dataset = arguments.dataset)