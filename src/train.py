#Imports (Double-check what's necessary here)

from datasets import load_dataset
from transformers import AutoTokenizer

from itertools import islice
from typing import Iterable, List, Tuple

import argparse
import numpy as np
import torch
import datasets
from datasets.dataset_dict import DatasetDict

from gensim.models.keyedvectors import KeyedVectors
from torch import nn
import random


## - Model training script

# What should this function take as input? What should be run elsewhere?

# - Preprocessed data, definitely. In the form of a huggingface dataset.
# - Validation and training sets.
# - Labels and dependent variable.
# - The already-loaded model, I assume?

def train_model(model, train_dataloader, val_dataloader, optimizer, epochs = False, model_name = "trained_model_nruns_"):

    nosave = 0
    #if epochs:
    print("[INFO:] Training classifier...")
    nruns = 0
    #if epochs: # If limited epochs is defined.
    while True:
        nruns += 1
        if epochs:
            if nruns > epochs:
                print("Max epochs reached!")
                break
        k = 0
        for batch in train_dataloader:
            k += 1
            if k < 10:
                print("This is working for the ", k, "th time!")
            ## train on one batch
            # FORWARD PASS
            y = model(**batch)
            # Calculate loss
            loss = y.loss
            # backpropagation
            loss.backward()
            # take step, reset
            optimizer.step()
            optimizer.zero_grad()
        # Validate model post epoch
        for val_data in val_dataloader: # Roundabout way of loading validation data. Unsure how else to feed it to the model.
            val_y = model(**val_data)
        # Calculate validation loss
        val_loss = val_y.loss
        # If best_val_loss exists, see if it needs updating.
        try:
            if best_val_loss > val_loss:
                best_val_loss = val_loss
               # save the model if it is the best so far
                torch.save(model, f"models/{model_name}.pt")
                nosave = 0
            else:
                # Increment nosave
                nosave += 1
           # Otherwise, create it and save the model.
        except NameError:
            best_val_loss = val_loss
            torch.save(model, f"models/{model_name}")
        # stop the training if patience is used up
        if nosave > 5:
            print("Patience is up!")
            break
            # some print to see that it is running
        #if (epoch + 1) % 10 == 0: # Comment or uncomment depending on how many epochs you want printed.
        print(f"epoch: {nruns}, loss = {val_loss:.4f}")
    print("[INFO:] Finished traning!")
