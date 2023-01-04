# Imports 
from datasets import load_dataset
from datasets import load_from_disk
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import argparse
import numpy as np
import torch
import datasets
from torch import nn
import random
from src.data import arg_inputs

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
# seed
random.seed(1337)

# Load your data here.
def test_all(old_model_name, filename):
    test_data = load_from_disk("src/preprocessed/" + filename)
    print(type(test_data))
    num_labels = [int(i) for i in filename.split('_') if i.isdigit()][0]
    # Load a model to be tested.
    model = torch.load("src/models/" + old_model_name)
    def test(model, test_data):
        output = model(input_ids = test_data["input_ids"], attention_mask = test_data["attention_mask"], labels = test_data["labels"])
        # Probs is a numpy-array with a lot of prob values.
        guesses = np.argmax(output.logits.softmax(dim = -1).detach().numpy(), axis = 1)
        return guesses.tolist()
    len_test = len(test_data)
    i = 0
    guesses = []
    # For script testing purposes, uncomment the following line:
    # len_test = 101
    while True:
        if i + 100 < len_test:
            small_test = test_data.select(range(i, i +100))
            guesses = guesses + test(model, small_test)
        else:
            small_test = test_data.select(range(i, len_test))
            guesses = guesses + test(model, small_test)
            break
        i += 100
    y_pred = guesses
    y_true = test_data["labels"]
    y_true=y_true.detach().numpy().tolist()
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, labels = list(range(num_labels)), average = 'micro')
    report = classification_report(y_true, y_pred)
    print("f1 score:", f1)
    print("accuracy score", accuracy)
    print(report)
    # # Save classification report
    file_path = r'src/out/' 
    file_name = file_path + f"classification_report_{old_model_name}.txt"
    text_file = open(file_name, 'w')
    my_string = report
    text_file.write(my_string)
    text_file.close()

# Command line functionality
if __name__ == "__main__":
    arguments = arg_inputs()
    test_all(old_model_name = arguments.old_model_name, filename = arguments.test_data)