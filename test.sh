# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 
python3 -m src.test --old_model_name "blabla.pt"  --test_data "gen_tldr_3_test"