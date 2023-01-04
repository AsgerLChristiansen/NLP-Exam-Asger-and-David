# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run for a single epoch and validate
python3 -m src.main --new_model_name "new_model_name.pt"  --train_data "training_set" --val_data "validation_set" --batch_size 64