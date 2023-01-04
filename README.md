# NLP-Exam-Asger-and-David

# Summary

The following repo was used for our Natural Language Processing Exam in the Cognitive Science MA degree, 1st semester, 2022.

This repo contains the following bash scripts:

## gen_tldr.sh
Downloads the reddit_tifu dataset from huggingface (https://huggingface.co/datasets/reddit_tifu) and creates artificial TL;DR's using the bert-extractive-summarizer package.
NB: This takes a VERY long time. As such, the dataset is split into 3 parts of ~10.000 examples. If a crash occurs, but one of these 3 parts were finished and succesfully, the script will start where it left off.
Requires no arguments and shouldn't be altered.
The TL;DR's will be stored in src/gen_tldr_temp - once all parts are finished and saved, one must then call preprocess.sh to preprocess them before analysis is possible.

## preprocess.sh
Should be run after gen_tldr.sh, but can also be run without it (assuming ) NOT be altered (i.e., takes no arguments). This downloads the reddit_tifu dataset (and, if available, the data stored in gen_tldr_temp is also loaded), tokenizes it (according to title, full body (docs), user-generated TL;DR and, if available, BERT-generated TL;DR's) and organizes it into train, test and validation splits in the src/preprocessed folder.

## run.sh
This bash script runs a single epoch of training and validation using available CPU (GPU functionality isn't included). As such, it takes a very long time (hence why only one epoch is run). Because of this method of running the program epoch-by-epoch, there is no patience parameter.
For this script to work, valid training and validation sets must exist in the src/preprocessed folder. This will be the case if preprocess.sh has been run ahead of time.
To run this script, it is necessary to alter variables in Line 6 manually (we did not have the time to set it up more conveniently and only one line needs to be changed).
To train a new model, alter Line 6 to look like this:

python3 -m src.main --new_model_name "new_model_name.pt"  --train_data "training_set" --val_data "validation_set" --batch_size 64

in which "training_set" and "validation_set" should be changed to appropriate names of folders in src/preprocessed (Do NOT write the filepath, only the folder names!), and "new_model_name.pt" is the name you want to save the model under (it will be saved to src/models).

To run a model for one more epoch, alter Line 6 to look like this:

python3 -m src.main --old_model_name "old_model_name.pt" --new_model_name "new_model_name.pt"  --train_data "training_set" --val_data "validation_set" --batch_size 64

in which "old_model_name.pt" is a model from src/models that you wish to train further, and "new_model_name.pt" is the name you want to save it under. We recommend specifying the number of epochs a model has been trained on in the filename. Also, keep in mind that if you overwrite a model, you can't test it later.

As can be seen, batch size can also be changed. Default is 64.

NOTHING ELSE THAN THESE VARIABLES should be changed about run.sh!

## test.sh
Similarly to run.sh, this model tests an already saved model on a specified test dataset. To run it, alter Line 6 to look as follows and change nothing else:

python3 -m src.test --old_model_name "old_model_name.pt"  --test_data "test_data"

in which "old_model_name.pt" is a model from src/models that you wish to test, and "test_data" is an appropriate dataset from src/preprocessed (Again, only specify folder name, not filepath).

Outputs a classification report to src/out

## Project Organization
This is the repo structure of the project:

```

├── README.md                  <- You're reading this! :)
├── src            
│   └── data.py                <- Script for processing arguments, downloading and preprocessing data.
│   └── gen_tldr.py            <- Script for generating TL;DR's
│   └── main.py                <- main test and validation script. Runs one epoch only.
│   └── test.py                <- test script for an existing model.
│   └── validate.py            <- validation script for an existing model.
│   └── train.py               <- training script, used on an existing model, or bert-base-uncased.
│   └── inspect_data.ipynb     <- Informally structured ipynb used for plotting and the like, unused by any bash scripts.
│   └── preprocessed           <- data storage folder.
│   └── models                 <- models storage folder.
│   └── gen_tldr_temp          <- temp storage for generated tldr before preprocessing
│   └── out                    <- plots and classification 
├── gen_tldr.sh                <- Top level bash script for TL;DR generation.
├── preprocess.sh              <- Top level bash script for preprocessing.
├── run.sh                     <- Top level bash script for training and re-training models, validating at every step.
├── test.sh                    <- Top level test script
├── requirements.txt           <- A requirements file of the required packages.
└── assignment_description.md  <- the assignment description

```
