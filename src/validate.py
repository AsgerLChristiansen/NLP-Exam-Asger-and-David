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

random.seed(1337)
# Import training script (and allow for changes to it)
from datasets import load_from_disk

# Load your data here.

val_cl_long = load_from_disk("preprocessed/gen_tldr_val")

# Load a model to be validated.
model = torch.load("models/tldr_model.pt")

from torch.utils.data import DataLoader

# The way we feed data to the model is using the dataloader class. 
val_cl_long = val_cl_long.remove_columns(['ups', 'num_comments', 'upvote_ratio', 'score', 'documents', 'tldr', 'title'])
val_cl_long.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
len_val = len(val_cl_long["input_ids"])

val_loss = []
i = 0
#len_val = 600
while True:
    if i + 320 < len_val:
        small_val = val_cl_long.select(range(i, i + 320))
        val_dataloader = torch.utils.data.DataLoader(small_val, batch_size = 32)
        print("bla")
        for val_data in val_dataloader: # Roundabout way of loading validation data. Unsure how else to feed it to the model.
            val_y = model(**val_data)
        # Calculate validation loss
            val_loss.append(val_y.loss)
        print("blabla")
        print(val_y.loss)
    else:
        small_val = val_cl_long.select(range(i, len_val))
        val_dataloader = torch.utils.data.DataLoader(small_val, batch_size = 32)
        for val_data in val_dataloader: # Roundabout way of loading validation data. Unsure how else to feed it to the model.
            val_y = model(**val_data)
            # Calculate validation loss
            val_loss.append(val_y.loss)
        break
    i += 320

val_loss = val_loss.detach().numpy()

print("Validation loss = ", np.mean(val_loss))