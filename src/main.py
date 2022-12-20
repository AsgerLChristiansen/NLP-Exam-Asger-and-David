# Run the setup script first! Migrate to the correct folder and type bash setup.sh in the terminal.

# Import loads of stuff (I don't think I even use half of these.)
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
import importlib
import train
importlib.reload(train)

# Let's train a model!
model_name = "bert-base-uncased"

from datasets import load_from_disk



# # Your preprocessed train, val and test sets.
train_cl_long = load_from_disk("preprocessed/train_long")
# val_cl_long = load_from_disk("preprocessed/val_long")
# test_cl_long = load_from_disk("preprocessed/test_long")

#train_cl_long = load_from_disk("preprocessed/gen_tldr_train")
from torch.utils.data import DataLoader

# The way we feed data to the model is using the dataloader class. It expects a very, very specific sort of input that looks like this:
train_cl_long = train_cl_long.remove_columns(['ups', 'num_comments', 'upvote_ratio', 'score', 'documents', 'tldr', 'title'])
train_cl_long.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
# And this:
#val_cl_long = val_cl_long.remove_columns(['ups', 'num_comments', 'upvote_ratio', 'score', 'documents', 'tldr', 'title'])
#val_cl_long.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])

# Your input ids are your tokens, labels are the upvote ratio.

len_train = len(train_cl_long["input_ids"])
#len_val = len(val_cl_long["input_ids"])

# Then go on holiday, run a marathon, write a book, get married, etc. while it runs.
#train_dataloader = torch.utils.data.DataLoader(train_cl_long, batch_size=32)
#val_dataloader = torch.utils.data.DataLoader(val_cl_long, batch_size = len(val_cl_long))

#Import model and optimizer
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=11) # We have 11 labels.
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)

## Train model for the very first time (if you want to load an already trained model
# to keep training it, use the code blpck below.)

## IMPORTANT! If you want the model to run until validation loss is minimized, epochs should just be set to False.
# In practice, however, these models take so long that I doubt we will ever get to the minimum,
# so specifying a number for epochs is probably a good idea.
#epochs = 1
model_name = "actual_tldr_model_epoch_1"  # Replace with whatever else you want. The train function ensures that it is a .pt model.

i = 0
while True:
    if i + 320 < len_train:
        small_train = train_cl_long.select(range(i, i +320))
        train_dataloader = torch.utils.data.DataLoader(small_train, batch_size=32)
        train.train_model(model, train_dataloader, optimizer, model_name)
    else:
        small_train = train_cl_long.select(range(i, len_train))
        train_dataloader = torch.utils.data.DataLoader(small_train, batch_size=32)
        train.train_model(model, train_dataloader, optimizer, model_name)
        break
    i += 320


#small_val = val_cl_long.select(range(100))
#val_dataloader = torch.utils.data.DataLoader(small_val, batch_size = len(small_val))