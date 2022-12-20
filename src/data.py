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

# Load datasets
tifu_short_raw = load_dataset("reddit_tifu", "short", split="train")
tifu_long_raw = load_dataset("reddit_tifu", "long", split = "train")

# Add decimal labels to upvote ratio
def add_columns(example):
    example["upvote_ratio"] = round(example["upvote_ratio"], 1)
    example["labels"] = int(example["upvote_ratio"] * 10)
    return example

tifu_short = tifu_short_raw.map(add_columns)

tifu_long = tifu_long_raw.map(add_columns)



# Split into train, validation and test.
dict1 = tifu_long.train_test_split(test_size=0.1)

train_and_val = dict1["train"]
test_long = dict1["test"]

dict2 = train_and_val.train_test_split(test_size=0.2)

train_long = dict2["train"]
val_long = dict2["test"]

# Some light preprocessing:

def remove_new_lines(example):
    text = example["documents"]
    example["documents"] = text.replace('\n\n', '. ').replace('\n', '. ') # Replace double and single new lines with dots.
    return example

train_long = train_long.map(remove_new_lines)
val_long = val_long.map(remove_new_lines)
test_long = test_long.map(remove_new_lines)

# Tokenization time!
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


## LOOK HERE! If you want to train on full text, run as it is. If you want to train on titles, change 'documents' to 'title'.
def tokenization(example): 
    return tokenizer(example["tldr"], padding  = 'max_length', truncation= True, return_tensors = "pt")

# Tokenize and save to preprocessed. IMPORTANT: Don't type "git add ." ever! These files are too large to add to git, and
# it becomes very problematic to change once you've added and committed them!!!!
train_input = train_long.map(tokenization, batched = True)
val_input = val_long.map(tokenization, batched = True)
test_input = test_long.map(tokenization, batched = True)

train_input.save_to_disk("preprocessed/train_long")
val_input.save_to_disk("preprocessed/val_long")
test_input.save_to_disk("preprocessed/test_long")


# Tokenization time!
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


from datasets import load_from_disk, concatenate_datasets

# Your preprocessed train, val and test sets.
one = load_from_disk("preprocessed/gen_tldr_1")
two = load_from_disk("preprocessed/gen_tldr_2")
three = load_from_disk("preprocessed/gen_tldr_3")

gen_tldr = concatenate_datasets([one,two,three])

## LOOK HERE! If you want to train on full text, run as it is. If you want to train on titles, change 'documents' to 'title'.
def tokenization(example): 
    return tokenizer(example["tldr"], padding  = 'max_length', truncation= True, return_tensors = "pt")

# Tokenize and save to preprocessed. IMPORTANT: Don't type "git add ." ever! These files are too large to add to git, and
# it becomes very problematic to change once you've added and committed them!!!!
gen_tldr_cleaned = gen_tldr.map(tokenization, batched = True)


# Split into train, validation and test.
dict1 = gen_tldr_cleaned.train_test_split(test_size=0.1)

train_and_val = dict1["train"]
test = dict1["test"]

dict2 = train_and_val.train_test_split(test_size=0.2)

train = dict2["train"]
val = dict2["test"]

train.save_to_disk("preprocessed/gen_tldr_train")
val.save_to_disk("preprocessed/gen_tldr_val")
test.save_to_disk("preprocessed/gen_tldr_test")


# Import training script (and allow for changes to it)
import importlib
import train
importlib.reload(train)

# Let's train a model!
model_name = "bert-base-uncased"

from datasets import load_from_disk

# Your preprocessed train, val and test sets.
train_cl_long = load_from_disk("preprocessed/train_long")
val_cl_long = load_from_disk("preprocessed/val_long")
test_cl_long = load_from_disk("preprocessed/test_long")


from torch.utils.data import DataLoader

# The way we feed data to the model is using the dataloader class. It expects a very, very specific sort of input that looks like this:
train_cl_long = train_cl_long.remove_columns(['ups', 'num_comments', 'upvote_ratio', 'score', 'documents', 'tldr', 'title'])
train_cl_long.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
# And this:
val_cl_long = val_cl_long.remove_columns(['ups', 'num_comments', 'upvote_ratio', 'score', 'documents', 'tldr', 'title'])
val_cl_long.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])

# Your input ids are your tokens, labels are the upvote ratio.


# This model takes a long-ass time to train even a single epoch on.
# These variables are here to ensure that you can test it on small samples.
small_train = train_cl_long.select(range(32))
small_val = val_cl_long.select(range(10))
#train_dataloader = torch.utils.data.DataLoader(small_train, batch_size=32)
#val_dataloader = torch.utils.data.DataLoader(small_val, batch_size = len(small_val))

# Then go on holiday, run a marathon, write a book, get married, etc. while it runs.
train_dataloader = torch.utils.data.DataLoader(train_cl_long, batch_size=32)
val_dataloader = torch.utils.data.DataLoader(val_cl_long, batch_size = len(val_cl_long))

#Import model and optimizer
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=11) # We have 11 labels.
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)