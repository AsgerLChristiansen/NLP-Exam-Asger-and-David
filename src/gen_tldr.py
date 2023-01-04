# Imports
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import argparse
import numpy as np
import torch
import datasets
from datasets import load_from_disk
from torch import nn
import random

# Filter dataset_short for rows without a tldr
tifu_short = load_dataset("reddit_tifu", "short", split="train")
tifu_no_tldr = tifu_short.filter(lambda example: example["tldr"] == "")

# Create smaller chunks of the no_tldr dataset
small_no_tldr_1 = tifu_no_tldr.select(list(range(10000)))
small_no_tldr_1 = tifu_no_tldr.select(list(range(10)))
small_no_tldr_2 = tifu_no_tldr.select(list(range(10000, 20000)))
small_no_tldr_3 = tifu_no_tldr.select(list(range(20000, len(tifu_no_tldr))))

# Generate tldr for 10.000 posts
from summarizer import Summarizer
bert_model = Summarizer()

# Add decimal labels to upvote ratio
def gen_tldr_func(example):
    raw_text = example["documents"]
    cleaner_text = raw_text.replace("\n\n", '.')
    clean_text = raw_text.replace('\n', '.')
    bert_summary = ''.join(bert_model(clean_text, num_sentences = 2))
    example["tldr"] = bert_summary
    return example

try:
    load_from_disk("src/gen_tldr_temp/gen_tldr_1")
    print("[NOTICE]")
    print("First 10.000 already generated.")
except FileNotFoundError:
    # Generate the 1st 10.000 tldr's.
    gen_tldr_1 = small_no_tldr_1.map(gen_tldr_func)
    gen_tldr_1.save_to_disk("src/gen_tldr_temp/gen_tldr_1")


try:
    load_from_disk("src/gen_tldr_temp/gen_tldr_2.")
    print("[NOTICE]")
    print("Second 10.000 already generated")
except FileNotFoundError:
    # Generate the 2nd 10.000 tldr's.
    gen_tldr_2 = small_no_tldr_2.map(gen_tldr_func)
    gen_tldr_2.save_to_disk("src/gen_tldr_temp/gen_tldr_2")


try:
    load_from_disk("src/gen_tldr_temp/gen_tldr_3")
    print("[NOTICE]")
    print("Third 10.000 plus remainder already generated.")
except FileNotFoundError:
# Generate the rest
    gen_tldr_3 = small_no_tldr_3.map(gen_tldr_func)
    gen_tldr_3.save_to_disk("src/gen_tldr_temp/gen_tldr_3")