#Then: train.py
#Training loop

#Evaluation script (CoNLL scorer)

#Optionally: support for gradient accumulation, early stopping, multi-GPU (Accelerate or DDP)

from transformers import XLMRobertaTokenizer
from dataset import CoreferenceDataset
from torch.utils.data import DataLoader

import torch
import time

from dataset import collate_fn
import model

start = time.time()
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
dataset = CoreferenceDataset("data/unc-gold-train-json/", tokenizer)
dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
print(f"Loaded tokenizer and dataset in {time.time() - start} seconds")

start = time.time()
xlmr_model = model.XLMRCorefModel()
print(f"Loaded XMLR model in {time.time() - start} seconds")

# Check if CUDA is available and move model to device (GPU if possible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xlmr_model.to(device)

# Test the model with a batch
for batch in dataloader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    # Forward pass through the model
    with torch.no_grad():  # No need to track gradients for this test
        embeddings = xlmr_model(input_ids, attention_mask)

    print(f"Embeddings shape: {embeddings.shape}")  # Should print: (batch_size, seq_len, hidden_size)
    break  # Only test one batch for now