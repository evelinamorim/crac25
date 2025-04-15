#Then: train.py
#Training loop

#Evaluation script (CoNLL scorer)

#Optionally: support for gradient accumulation, early stopping, multi-GPU (Accelerate or DDP)

from transformers import XLMRobertaTokenizerFast
from dataset import CoreferenceDataset
from torch.utils.data import DataLoader

from constants import DEPENDENCY_LABELS

import torch
import time

from dataset import collate_fn
import model

start = time.time()
tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")
dataset = CoreferenceDataset("data/unc-gold-train-json/", tokenizer)
dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
print(f"Loaded tokenizer and dataset in {time.time() - start} seconds")

start = time.time()

# TOD: determine a confif file with:
config = {}
config["span_width"] = 30
config["combine_strategy"] = "concat"

xlmr_model = model.XLMRCorefModel(config, num_labels=len(DEPENDENCY_LABELS)) # combine_strategy = add or concat
print(f"Loaded XMLR model in {time.time() - start} seconds")

# Check if CUDA is available and move model to device (GPU if possible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xlmr_model.to(device)

# Test the model with a batch
for batch in dataloader:

    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    edges = batch['edges']

    # Convert edge tensors to integers and detach from graph
    processed_edges = [
        [(s.item(), t.item(), l.item()) for (s, t, l) in edge_list]
        for edge_list in edges
    ]
    span_starts = [s.to(device) for s in batch["span_starts"]]
    span_ends = [e.to(device) for e in batch["span_ends"]]
    cluster_ids = [c.to(device) for c in batch["cluster_ids"]]
    # Forward pass through the model
    with torch.no_grad():  # No need to track gradients for this test
         #embeddings, span_reps, mention_scores
         output = xlmr_model(input_ids, attention_mask, edges=processed_edges,
                                span_starts=span_starts, span_ends=span_ends)

    #print(f"Embeddings: {embeddings}")  # Should print: (batch_size, seq_len, hidden_size)
    break  # Only test one batch for now