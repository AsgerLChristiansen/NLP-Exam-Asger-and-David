#Imports (Double-check what's necessary here)

from datasets import load_dataset
from transformers import AutoTokenizer

import argparse
import numpy as np
import torch
import datasets
from datasets.dataset_dict import DatasetDict
from torch import nn
import random


## - Model training script
def train_model(model, train_dataloader, optimizer, model_name = "trained_model_"):
    for batch in train_dataloader:
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
    torch.save(model, f"src/models/{model_name}")
    # Otherwise, create it and save the model.
    # some print to see that it is running
