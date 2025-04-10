import json
import os
import torch
from torch.utils.data import Dataset

from collections import defaultdict

from constants import DEPENDENCY_LABELS


class CoreferenceDataset(Dataset):
    def __init__(self, dir_path, tokenizer, max_length=512):
        self.data = self.load_data(dir_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = {label: idx for idx, label in enumerate(DEPENDENCY_LABELS)}

    def load_data(self, dir_path):

        file_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
                      if f.endswith('.json')]
        combined_data = []
        for path in file_paths:
            with open(path, 'r') as f:
                data = json.load(f)
                combined_data.extend(data)
        return combined_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        # Tokenization and alignment
        tokenized = self.tokenizer(
            example["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt"
        )

        # Create word to subword mapping
        word_ids = tokenized.word_ids()
        word_to_subwords = defaultdict(list)
        for subword_idx, word_id in enumerate(word_ids):
            if word_id is not None:
                word_to_subwords[word_id].append(subword_idx)

        # Process coreference clusters
        clusters = defaultdict(list)
        for mention in example["mentions"]:
            start_word = mention["start"]
            end_word = mention["end"]
            cluster_id = mention["cluster"]

            # Convert word spans to subword spans
            subword_start = word_to_subwords[start_word][0] if start_word in word_to_subwords else -1
            subword_end = word_to_subwords[end_word - 1][-1] if (end_word - 1) in word_to_subwords else -1

            if subword_start != -1 and subword_end != -1:
                clusters[cluster_id].append((subword_start, subword_end))

        # Process dependency edges
        edge_list = []
        for edge in example["edges"]:
            source = edge["source"]
            target = edge["target"]
            label = edge["label"]

            # Convert edge labels to IDs if needed
            edge_list.append((
                torch.tensor(source, dtype=torch.long),
                torch.tensor(target, dtype=torch.long),
                torch.tensor(self.label2id.get(label, 0))  # Handle unknown labels
                             ))

        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "clusters": clusters,
            "edges": edge_list,
            "lang": example["lang"]
        }


def collate_fn(batch):
    padded_batch = {
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=1  # XLMR pad token
        ),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(
            [item["attention_mask"] for item in batch],
            batch_first=True,
            padding_value=0
        ),
        "clusters": [item["clusters"] for item in batch],
        "edges": [item["edges"] for item in batch],
        "langs": [item["lang"] for item in batch]
    }
    return padded_batch



