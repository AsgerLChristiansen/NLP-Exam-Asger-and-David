#Imports (Double-check what's necessary here)

from datasets import load_dataset
from transformers import AutoTokenizer

#Import model and optimizer
from transformers import AutoModelForSequenceClassification

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

val_data= load_from_disk("preprocessed/tldr_val")

# Load a model to be validated.
model = torch.load("models/3_tldr_epoch_1.pt")

# bert_model = "bert-base-uncased"
# model = AutoModelForSequenceClassification.from_pretrained(bert_model, num_labels=11)
#from torch.utils.data import DataLoader

def validate(val_data):
    output = model(input_ids = val_data["input_ids"], attention_mask = val_data["attention_mask"], labels = val_data["labels"])
    loss = output.loss
    print(loss.item())
    return loss.item()

len_val = len(val_data)

#len_val = 1000
val_loss = []
i = 0
while True:
    if i + 100 < len_val:
        small_val = val_data.select(range(i, i +100))
        loss= validate(small_val)
        val_loss.append(loss)
    else:
        small_val = val_data.select(range(i, len_val))
        validate(small_val)
        loss = validate(small_val)
        val_loss.append(loss)
        break
    i += 100

print("Mean validation loss: ", np.mean(val_loss))