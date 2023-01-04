# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Test model
python3 -m src.test --old_model_name "old_model_name.pt"  --test_data "test_data"