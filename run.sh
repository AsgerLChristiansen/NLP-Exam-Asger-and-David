# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 
python3 -m src.main --new_model_name "blabla.pt"  --train_data "tldr_3_train" --val_data "tldr_3_val" --batch_size 64