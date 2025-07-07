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
                combined_data.append(data)
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
        numeric_clusters = defaultdict(list)
        for cid, cluster in enumerate(example["clusters"]):
            for start_word, end_word in cluster:

                subword_start = word_to_subwords[start_word][0] if (start_word in word_to_subwords) else -1
                subword_end = word_to_subwords[end_word][-1] if (end_word in word_to_subwords) else -1

                if subword_start != -1 and subword_end != -1:
                    numeric_clusters[cid].append((subword_start, subword_end))

        # Process dependency edges
        edge_list = []
        for edge in example["edges"]:
            source_word = edge["source"]
            target_word = edge["target"]
            label = edge["label"]

            if source_word in word_to_subwords and target_word in word_to_subwords:
                source_subword = word_to_subwords[source_word][0]  # First subword of source
                target_subword = word_to_subwords[target_word][0]  # First subword of target

                edge_list.append((
                    torch.tensor(source_subword, dtype=torch.long),
                    torch.tensor(target_subword, dtype=torch.long),
                    torch.tensor(self.label2id.get(label, 0))
                ))

        span_starts = []
        span_ends = []
        cluster_ids = []

        for cid, mentions in numeric_clusters.items():
            for (s, e) in mentions:
                span_starts.append(s)
                span_ends.append(e)
                cluster_ids.append(cid)

        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "clusters": numeric_clusters,
            "edges": edge_list,
            "span_starts": torch.tensor(span_starts),
            "span_ends": torch.tensor(span_ends),
            "cluster_ids": torch.tensor(cluster_ids),
            "sentence_map": torch.tensor(example.get("sentence_map")).squeeze(0),
            "sentence_starts": torch.tensor(example.get("sentence_starts")),
            "lang": example["lang"]
        }

from torch.nn.utils.rnn import pad_sequence
import torch

def collate_fn(batch):
    # Convert edges, spans, and cluster_ids to tensors and pad them
    processed = {
        "input_ids": pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=1  # XLM-RoBERTa pad token ID
        ),
        "attention_mask": pad_sequence(
            [item["attention_mask"] for item in batch],
            batch_first=True,
            padding_value=0
        ),
        # Convert edges to tensors and pad
        "edges": pad_sequence(
            [torch.tensor(item["edges"], dtype=torch.long) for item in batch],
            batch_first=True,
            padding_value=-100  # Use a padding value your model can ignore
        ),
        # Convert span starts/ends to tensors and pad
        "span_starts": pad_sequence(
            [item["span_starts"] for item in batch],
            batch_first=True,
            padding_value=-100
        ),
        "span_ends": pad_sequence(
            [item["span_ends"] for item in batch],
            batch_first=True,
            padding_value=-100
        ),
        # Convert cluster IDs to tensors and pad
        "cluster_ids": pad_sequence(
            [item["cluster_ids"] for item in batch],
            batch_first=True,
            padding_value=-100
        ),
        "sentence_map": pad_sequence(
            [item["sentence_map"] for item in batch],
            batch_first=True,
            padding_value=0
        ),
        "sentence_starts":pad_sequence(
            [item["sentence_starts"] for item in batch],
            batch_first=True,
            padding_value=0
        ),
        # Non-tensor fields (keep as lists)
        "clusters": [item["clusters"] for item in batch],
        "langs": [item["lang"] for item in batch]
    }

    return processed



