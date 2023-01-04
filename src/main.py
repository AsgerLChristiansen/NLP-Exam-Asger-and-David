# Imports
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse
import numpy as np
import torch
import datasets
from torch import nn
import random
from src.data import arg_inputs
from src.train import train_model
from src.validate import validate
random.seed(1337)

# Import training script (and allow for changes to it)
import importlib

#Import model and optimizer
from transformers import AutoModelForSequenceClassification

from torch.optim import AdamW
from datasets import load_from_disk

from torch.utils.data import DataLoader

def main(old_model_name:str = None, new_model_name:str = "My model.pt", train_data:str = "docs_3_train", val_data = "docs_3_val", batch_size = 64):
    """_summary_

    Args:
        old_model_name (str): Name of model to be loaded. Remember .pt extension. Use only if training an existing model.
        new_model_name (str): Name that model should be saved under. Recommend including epoch number in here!
        train_data (str): name of folder w/ training data from src/preprocessed
        val_data (str): name of folder w/ validation data from src/preprocessed
        batch_size (int): Batch size. Defaults to 64. Should ideally be something that can divide 320.
    """
    num_labels = [int(i) for i in train_data.split('_') if i.isdigit()][0]
    train = load_from_disk("src/preprocessed/" + train_data)
    val = load_from_disk("src/preprocessed/" + val_data)
    # Get length of input
    len_train = len(train["input_ids"])
    len_val = len(val["input_ids"])
    if old_model_name:
        model = torch.load('src/models/' + old_model_name)
    else:
        # Let's train a model!
        bert_model = "bert-base-uncased"
        model = AutoModelForSequenceClassification.from_pretrained(bert_model, num_labels=num_labels) # We have 11 labels.
    optimizer = AdamW(model.parameters(), lr=5e-5)
    # Training loop (Batch size of 32. Data is subsetted into chunks of 320 to avoid crashing issues)   
    # Runs only one epoch due to technical limitations. LEss likely to crash this way.
    i = 0
    j = 0
    print("Starting training...")    # For script testing purposes, uncomment the following line:
    # len_train = 321
    while True:
        if i + 320 < len_train:
            small_train = train.select(range(i, i +320))
            train_dataloader = torch.utils.data.DataLoader(small_train, batch_size=batch_size)
            train_model(model, train_dataloader, optimizer, new_model_name)
            print("Bla")
        else:
            small_train = train.select(range(i, len_train))
            train_dataloader = torch.utils.data.DataLoader(small_train, batch_size=batch_size)
            train_model(model, train_dataloader, optimizer, new_model_name)
            break
        i += 320
        j += 1
        if j % 10 == 0:
            print("100 batches processed!")
    print("Training finished, validating in subsets of 100 examples (To avoid crashing)...")
    
    # For script testing purposes, uncomment the following line:
    # len_val = 101
    val_loss = []
    i = 0
    while True:
        if i + 100 < len_val:
            small_val = val.select(range(i, i +100))
            loss= validate(small_val, model)
            val_loss.append(loss)
        else:
            small_val = val.select(range(i, len_val))
            loss = validate(small_val, model)
            val_loss.append(loss)
            break
        i += 100
    print("Validation finished!")
    print("Mean validation loss: ", np.mean(val_loss))

# Command line functionality
if __name__ == "__main__":
    arguments = arg_inputs()
    main(old_model_name = arguments.old_model_name, new_model_name = arguments.new_model_name, train_data = arguments.train_data, val_data = arguments.val_data, batch_size = arguments.batch_size)