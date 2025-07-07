import os
from transformers import XLMRobertaTokenizerFast, get_linear_schedule_with_warmup
import time
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import CoreferenceDataset, collate_fn
import model
from constants import DEPENDENCY_LABELS

from coval.eval.evaluator import evaluate_documents


tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")

start = time.time()
dataset = CoreferenceDataset("data/sample-train/", tokenizer)
dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)


val_dataset = CoreferenceDataset("data/sample-dev/", tokenizer)
val_dataloader = DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn)
print(f"Loaded train and validation dataset in {time.time() - start} seconds")


# TOD: determine a confif file with:
config = {}
config["span_width"] = 10
config["combine_strategy"] = "concat"
num_epochs = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
xlmr_model = model.XLMRCorefModel(config, num_labels=len(DEPENDENCY_LABELS)).to(device) # combine_strategy = add or concat
optimizer = optim.AdamW(xlmr_model.parameters(), lr=1e-5)

total_steps = len(dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps
)
os.makedirs("checkpoints", exist_ok=True)

do_train = True

if do_train:
    xlmr_model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch} / {num_epochs}")
        total_train_loss = 0
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            optimizer.zero_grad()
            loss = xlmr_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                edges=batch['edges'],
                span_starts=batch['span_starts'],
                span_ends=batch['span_ends'],
                cluster_ids=batch['cluster_ids'],
                sentence_map=batch['sentence_map'],
                sentence_starts=batch['sentence_starts']
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(xlmr_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            print(f"Loss: {loss.item():.4f}")

            torch.save({
                'epoch': epoch,
                'model_state_dict': xlmr_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, f'checkpoints/xlmr_coref_checkpoint.pt')

#        avg_train_loss = total_train_loss / len(dataloader)
#        print(f"  Average training loss: {avg_train_loss:.4f}")
    # save the last model

#else:
#    start = time.time()
#    checkpoint_path = os.path.join("checkpoints","xlmr_coref_checkpoint_0.pt")
#    checkpoint = torch.load(checkpoint_path)
#    print(f"Loaded checkpoint from {checkpoint_path} in {time.time() - start}s...")


#    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluation")):
#        batch = next(iter(val_dataloader))
#        input_ids = batch['input_ids'].to(device)
#        attention_mask = batch['attention_mask'].to(device)
#        edges = batch['edges']  # Usually a list — not on device
#        span_starts = [s.to(device) for s in batch['span_starts']]
#        span_ends = [e.to(device) for e in batch['span_ends']]
#        clusters = xlmr_model.predict(input_ids, attention_mask, edges)







