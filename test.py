import model
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from dataset import CoreferenceDataset, collate_fn
from transformers import XLMRobertaTokenizerFast

from constants import DEPENDENCY_LABELS
# TOD: determine a confif file with:
config = {}
config["span_width"] = 10
config["combine_strategy"] = "concat"

model = model.XLMRCorefModel(config,num_labels=len(DEPENDENCY_LABELS), training=False)
checkpoint = torch.load('checkpoints/xlmr_coref_checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])

tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")
val_dataset = CoreferenceDataset("data/sample-dev/", tokenizer)
val_dataloader = DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn)

for batch_idx, batch in enumerate(tqdm(val_dataloader, desc="Validation")):
    # Run inference
    predictions = model.predict_batch(batch, tokenizer, threshold=0.48)

    # Process results

    for el in predictions:
        print(f"Cluster {el['cluster_id']}")
        for mention in el['mentions_txt']:
            print(f"Mention: {mention['text']}")
        print()