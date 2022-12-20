# Run the setup script first! Migrate to the correct folder and type bash setup.sh in the terminal.

# Import loads of stuff (I don't think I even use half of these.)
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse
import numpy as np
import torch
import datasets
from datasets.dataset_dict import DatasetDict
from torch import nn
import random

random.seed(1337)


def arg_inputs():
    """
    Initialize arguments for main() when called on command line

    """
    # initialise parser
    parser = argparse.ArgumentParser(description = "Script to run LSTM for Named Entity Recognition.")

    # add arguments
    parser.add_argument("-old_name",
                        "--old_model_name", 
                        type = str,
                        required = False,
                        help = "the name of an existing saved model from models/ to be trained on another epoch. Leave blank if training a model for the first time.")
    parser.add_argument("-new_name", 
                        "--new_model_name", 
                        type = str,
                        required = True,
                        help = "the name to be used when saving the model again after training.")
    parser.add_argument("-data",
                        "--dataset",
                        type = str,
                        required = True,
                        help = "the dataset from preprocessed/ to be trained on.")
    # list of arguments given
    args = parser.parse_args()

    # return list of arguments
    return args

# Load datasets from huggingface
def load_data():
    tifu_short_raw = load_dataset("reddit_tifu", "short", split="train")
    tifu_long_raw = load_dataset("reddit_tifu", "long", split = "train")
    return tifu_short_raw, tifu_long_raw

## FUNCTIONS
# Add decimal labels to upvote ratio
def add_columns(example):
    example["upvote_ratio"] = round(example["upvote_ratio"], 1)
    example["labels"] = int(example["upvote_ratio"] * 10)
    return example

# Remove new lines (Not really necessary in titles or tldr)
def remove_new_lines(example):
    text = example["documents"]
    example["documents"] = text.replace('\n\n', '. ').replace('\n', '. ') # Replace double and single new lines with dots.
    return example

# Tokenization
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

## LOOK HERE! If you want to train on full text, run as it is. If you want to train on titles, change 'documents' to 'title'.
def tokenization_tldr(example):
    return tokenizer(example["tldr"], padding  = 'max_length', truncation= True, return_tensors = "pt")

def tokenization_docs(example):
    return tokenizer(example["documents"], padding  = 'max_length', truncation= True, return_tensors = "pt")

def tokenization_title(example):
    return tokenizer(example["title"], padding  = 'max_length', truncation= True, return_tensors = "pt")


def clean_columns(dataset):
    dataset = dataset.remove_columns(['ups', 'num_comments', 'upvote_ratio', 'score', 'documents', 'tldr', 'title'])
    dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
    return dataset

def train_val_test(dataset, filename):
# Split into train, validation and test.
    dict1 = dataset.train_test_split(test_size=0.1)
    train_and_val = dict1["train"]
    test = dict1["test"]
    dict2 = train_and_val.train_test_split(test_size=0.2)
    train = dict2["train"]
    val = dict2["test"]
    train.save_to_disk("src/preprocessed/" + f'{filename}' + '_train.pt')
    val.save_to_disk("src/preprocessed/"+ f'{filename}' + '_val.pt')
    test.save_to_disk("src/preprocessed/"+ f'{filename}' + '_test.pt')
    return train, val, test


def preprocess(dataset, tokenization_func, filename):
    dataset = dataset.map(add_columns)
    dataset = dataset.map(remove_new_lines)
    dataset = dataset.map(tokenization_func, batched = True)
    dataset = clean_columns(dataset)
    train, val, test = train_val_test(dataset, filename)
    return None

if __name__ == "__main__":
    short, long = load_data()
    preprocess(short, tokenization_title, "title")
    preprocess(short, tokenization_docs, "docs")
    preprocess(short, tokenization_tldr, "tldr")