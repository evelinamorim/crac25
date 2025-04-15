#This will include:

#XLM-R for contextual embeddings

#GCN over syntactic edges (from UD)

#Span representation module (e.g., endpoint + attention)

#Mention scorer and clusterer

#If you’re replicating SpanBERT-style architecture, you’ll likely use mention-pair scoring or span clustering.

# model.py

import torch
from torch import nn
from transformers import XLMRobertaModel, XLMRobertaTokenizer



class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_labels):
        super(GCNLayer, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_labels)])
        self.relu = nn.ReLU()

    def forward(self, node_repr, edges):
        """
        node_repr: Tensor (batch_size, seq_len, hidden_size)
        edges: list of (source_idx, target_idx, label_id)
        """
        batch_size, seq_len, hidden_dim = node_repr.size()
        device = node_repr.device

        out = torch.zeros_like(node_repr)

        for b in range(batch_size):
            for source, target, label in edges[b]:
                out[b, target] += self.linear[label](node_repr[b, source])

        return self.relu(out)

class XLMRCorefModel(nn.Module):
    def __init__(self,config, hidden_size=768, gcn_hidden_size=768, num_labels=50):
        super(XLMRCorefModel, self).__init__()
        # Load the pre-trained XLM-RoBERTa model
        self.xlm_roberta = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.gcn = GCNLayer(hidden_size, gcn_hidden_size, num_labels)

        combine_strategy = config["combine_strategy"]
        self.span_width = config["span_width"]
        self.combine_strategy = combine_strategy

        self.span_repr = SpanRepresentation(hidden_size)
        self.mention_scorer = MentionScorer(hidden_size * 3)  # start + end + attn

        if combine_strategy == "concat":
            self.comb_proj = nn.Linear(hidden_size + gcn_hidden_size, hidden_size)

        self.antecedent_scorer = nn.Linear(hidden_size * 6, 1)
        self.dropout = nn.Dropout(0.1)
        self.span_width_emb = nn.Embedding(self.span_width, hidden_size)

    def generate_candidate_spans(self, embeddings):
        batch_size, seq_len, _ = embeddings.size()
        return [
            [(i, j) for i in range(seq_len)
             for j in range(i, min(i + self.span_width, seq_len))]
            for _ in range(batch_size)
        ]

    def prune_spans(self, span_reps, mention_scores, topk_per_doc):
        """
        Args:
            span_reps: list of (num_spans_i, span_dim) tensors
            mention_scores: list of (num_spans_i,) tensors
            topk_per_doc: tensor of shape (batch_size,) with topk per doc
        """
        pruned_span_reps = []
        pruned_scores = []
        pruned_indices = []

        for i, (reps, scores) in enumerate(zip(span_reps, mention_scores)):
            k = min(topk_per_doc[i].item(), scores.size(0))
            topk_scores, topk_indices = torch.topk(scores, k)
            pruned_span_reps.append(reps[topk_indices])
            pruned_scores.append(topk_scores)
            pruned_indices.append(topk_indices)

        return pruned_span_reps, pruned_scores, pruned_indices

    def get_antecedent_scores(self, span_reps):
        """Score all pairs of spans"""
        # Expand to all pairwise combinations
        num_spans = span_reps.size(0)
        cated = torch.cat([span_reps.unsqueeze(0).expand(num_spans, -1, -1),
                           span_reps.unsqueeze(1).expand(-1, num_spans, -1)], dim=-1)

        return self.antecedent_scorer(cated).squeeze(-1)

    def get_span_width_features(self, starts, ends):
        return self.span_width_emb(ends - starts)

    def forward(self, input_ids, attention_mask=None,edges=None, span_starts=None, span_ends=None):
        """
        Forward pass through XLM-RoBERTa model
        Args:
            input_ids: Tensor of tokenized input
            attention_mask: Tensor of attention mask (optional)
            edges: list of edge lists, one per batch element
        Returns:
            output: Hidden states (embeddings) from XLM-RoBERTa model
        """
        output = self.xlm_roberta(input_ids=input_ids, attention_mask=attention_mask) # (batch_size, seq_len, hidden_size)

        if edges is not None:
            gcn_out = self.gcn(output.last_hidden_state, edges)

            if self.combine_strategy == "add":
                embeddings = output.last_hidden_state + gcn_out
            elif self.combine_strategy == "concat":
                embeddings = self.comb_proj(torch.cat([output.last_hidden_state, gcn_out], dim=-1))
            else:
                raise ValueError("Invalid combine_strategy. Choose 'add' or 'concat'.")

            # Generate candidate spans (include gold mentions!)
            all_spans = self.generate_candidate_spans(embeddings)
            # Get span representations
            span_reps = self.span_repr(embeddings, all_spans)
            # Calculate mention scores
            mention_scores = self.mention_scorer(span_reps)
            seq_lens = attention_mask.sum(dim=1)
            topk_per_doc = (seq_lens.float() * 0.4).long()

            pruned_span_reps, pruned_scores, pruned_indices = self.prune_spans(
                span_reps, mention_scores, topk_per_doc
            )

            antecedent_scores_list = []

            for i in range(len(pruned_span_reps)):
                pruned_span_rep = pruned_span_reps[i]  # (topk, 3 * hidden)
                ant_scores = self.get_antecedent_scores(pruned_span_rep)  # (topk, topk)

                triu_mask = torch.triu(torch.ones_like(ant_scores,device=ant_scores.device), diagonal=0)

                ant_scores = ant_scores.masked_fill(triu_mask.bool(), float('-inf'))
                ant_scores = torch.nn.functional.log_softmax(ant_scores, dim=-1)

                antecedent_scores_list.append(ant_scores)

            normalized_scores = [torch.softmax(scores, dim=0) for scores in pruned_scores]

            #gold_spans = self.extract_gold_spans(clusters)  # From your labeled data

            # Combine candidates + gold spans
            #training_spans = list(set(all_spans + gold_spans))



        return output.last_hidden_state

class SpanRepresentation(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, embeddings, all_spans):
        """
        Args:
            embeddings: (batch_size, seq_len, hidden_size)
            all_spans: list of list of (start, end) tuples per example
        Returns:
            span_representations: list of tensors, one per example (num_spans_i, 3 * hidden)
        """
        span_reps = []

        for i, spans in enumerate(all_spans):  # Iterate over batch
            ex_span_reps = []
            for start, end in spans:
                span_emb = embeddings[i, start:end+1]  # (span_len, hidden)
                start_emb = embeddings[i, start]       # (hidden)
                end_emb = embeddings[i, end]           # (hidden)

                # Attention-based pooling over the span
                attn_scores = self.attention(span_emb)            # (span_len, 1)
                attn_weights = torch.softmax(attn_scores, dim=0)  # (span_len, 1)
                attn_output = torch.sum(attn_weights * span_emb, dim=0)  # (hidden)

                span_vec = torch.cat([start_emb, end_emb, attn_output], dim=-1)  # (3 * hidden)
                ex_span_reps.append(span_vec)

            span_reps.append(torch.stack(ex_span_reps))  # (num_spans, 3 * hidden)

        return span_reps


class MentionScorer(nn.Module):
    def __init__(self, span_repr_size):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(span_repr_size, span_repr_size),
            nn.ReLU(),
            nn.Linear(span_repr_size, 1)
        )

    def forward(self, span_reps):
        mention_scores = [self.scorer(rep).squeeze(-1) for rep in span_reps]
        return mention_scores

