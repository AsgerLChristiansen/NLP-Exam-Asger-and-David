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
from torch import nn
import random


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

random.seed(1337)

# Import training script (and allow for changes to it)
from datasets import load_from_disk

# Load your data here.

test_data= load_from_disk("preprocessed/gen_tldr_test_11")

model_name = "11_gen_tldr_epoch_1.pt"

# Load a model to be validated.
model = torch.load("models/" + model_name)

#bert_model = "bert-base-uncased"
#model = AutoModelForSequenceClassification.from_pretrained(bert_model, num_labels=11)
#from torch.utils.data import DataLoader

def test(test_data):
    output = model(input_ids = test_data["input_ids"], attention_mask = test_data["attention_mask"], labels = test_data["labels"])
    # Probs is a numpy-array with a lot of prob values.
    guesses = np.argmax(output.logits.softmax(dim = -1).detach().numpy(), axis = 1)
    return guesses.tolist()


len_test = len(test_data)

i = 0
guesses = []
while True:
    if i + 100 < len_test:
        small_test = test_data.select(range(i, i +100))
        guesses = guesses + test(small_test)
    else:
        small_test = test_data.select(range(i, len_test))
        guesses = guesses + test(small_test)
        break
    i += 100
y_pred = guesses

y_true = test_data["labels"]
y_true=y_true.detach().numpy().tolist()


accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, labels = list(range(11)), average = 'micro')
report = classification_report(y_true, y_pred)
print("f1 score:", f1)
print("accuracy score", accuracy)
print(report)
 # # Save classification report
file_path = r'out/' 
file_name = file_path + f"classification_report_{model_name}.txt"
text_file = open(file_name, 'w')
my_string = report
text_file.write(my_string)
text_file.close()