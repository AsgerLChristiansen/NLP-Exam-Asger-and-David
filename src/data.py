# Import loads of stuff (I don't think I even use half of these.)
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import argparse
import numpy as np
import torch
import datasets
from datasets import load_from_disk
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
                        required = False,
                        help = "the name to be used when saving the model again after training.")
    parser.add_argument("-train",
                        "--train_data",
                        type = str,
                        required = False,
                        help = "the dataset from preprocessed/ to be trained on.")
    parser.add_argument("-val",
                        "--val_data",
                        type = str,
                        required = False,
                        help = "the dataset from preprocessed/ to be validated on.")
    parser.add_argument("-batch",
                        "--batch_size",
                        type = int,
                        required = False,
                        help = "batch size to train on. Default 32")
    parser.add_argument("-test",
                        "--test_data",
                        type = str,
                        required = False,
                        help = "test data to test old models with.")
    
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

def reconfigure_labels(example):
    if example["labels"] in [0,1,2,3,4,5,6]:
        example["labels"] = 0
    elif example["labels"] in [7,8]:
        example["labels"] = 1
    else:
        example["labels"] = 2
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
    train.save_to_disk("src/preprocessed/" + f'{filename}' + '_train')
    val.save_to_disk("src/preprocessed/"+ f'{filename}' + '_val')
    test.save_to_disk("src/preprocessed/"+ f'{filename}' + '_test')
    return train, val, test


def preprocess(dataset, tokenization_func, filename, change_labels = False):
    dataset = dataset.map(add_columns)
    if change_labels == True:
        dataset = dataset.map(reconfigure_labels)
    dataset = dataset.map(remove_new_lines)
    dataset = dataset.map(tokenization_func, batched = True)
    dataset = clean_columns(dataset)
    train, val, test = train_val_test(dataset, filename)
    return None

if __name__ == "__main__":
    short, long = load_data()
    preprocess(short, tokenization_title, "title_11")
    preprocess(short, tokenization_docs, "docs_11")
    preprocess(short, tokenization_tldr, "tldr_11")
    preprocess(short, tokenization_title, "title_3", change_labels = True)
    preprocess(short, tokenization_docs, "docs_3", change_labels = True)
    preprocess(short, tokenization_tldr, "tldr_3", change_labels = True)
    try:
        one = load_from_disk("src/gen_tldr_temp/gen_tldr_1")
        two = load_from_disk("src/gen_tldr_temp/gen_tldr_2")
        three = load_from_disk("src/gen_tldr_temp/gen_tldr_3")
        # Make concatenated dataset
        gen_tldr = concatenate_datasets([one,two, three])
        # Tokenization
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        preprocess(gen_tldr, tokenization_tldr, "gen_tldr_11")
        preprocess(gen_tldr, tokenization_tldr, "gen_tldr_3", change_labels = True)
    except FileNotFoundError:
        print("[NOTICE]")
        print("It seems there are no generated tldr's to preprocess. If you have the time and computational resources, consider running gen_tldr.sh from the terminal.")