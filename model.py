#This will include:
from collections import defaultdict

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
        super().__init__()
        self.num_labels = num_labels
        self.linear = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_labels)])
        self.relu = nn.ReLU()

    def forward(self, node_repr, edges):
        batch_size, seq_len, hidden_dim = node_repr.shape
        device = node_repr.device

        mask = (edges[..., 2] != -100) & \
               (edges[..., 0] < seq_len) & \
               (edges[..., 1] < seq_len)  # (batch_size, max_edges)

        valid_edges = edges[mask]  # (total_valid_edges, 3)
        if valid_edges.size(0) == 0:
            return self.relu(torch.zeros_like(node_repr))

        batch_indices = torch.arange(batch_size, device=node_repr.device)
        batch_indices = batch_indices.repeat_interleave(mask.sum(dim=1))
        sources = valid_edges[:, 0].long()  # (total_valid_edges,)
        targets = valid_edges[:, 1].long()
        labels = valid_edges[:, 2].long()  # (total_valid_edges,)

        source_embs = node_repr[batch_indices, sources]  # (total_valid_edges, hidden_dim)
        transformed = torch.stack([self.linear[l](e) for l, e in zip(labels, source_embs)])

        out = torch.zeros_like(node_repr)
        flat_indices = batch_indices * seq_len + targets

        out_flat = out.view(-1, hidden_dim)
        out_flat.index_add_(
            dim=0,
            index=flat_indices,
            source=transformed
        )

        out = out_flat.view(batch_size, seq_len, hidden_dim)
        return self.relu(out)


def decode_clusters(pruned_spans, antecedent_scores):
    """
    Args:
        pruned_spans: list of (start, end) tuples, e.g. [(0,1), (4,4), (10,11)]
        antecedent_scores: Tensor of shape [num_spans, num_antecedents]
            where num_antecedents = i for each i-th span.

    Returns:
        clusters: List of coreference clusters (each is a list of (start, end))
    """
    num_spans = len(pruned_spans)
    predicted_clusters = []
    span_to_cluster = {}

    for i in range(num_spans):
        if i == 0:
            predicted_antecedent = -1  # No valid antecedents
        else:
            # Add dummy antecedent with score 0
            dummy_score = torch.tensor([0.0], device=antecedent_scores.device)
            scores = torch.cat([dummy_score, antecedent_scores[i][:i]], dim=0)
            predicted_antecedent = torch.argmax(scores).item() - 1

        current_span = pruned_spans[i]

        if predicted_antecedent == -1:
            # New cluster
            cluster_id = len(predicted_clusters)
            predicted_clusters.append([current_span])
            span_to_cluster[current_span] = cluster_id
        else:
            ant_span = pruned_spans[predicted_antecedent]
            cluster_id = span_to_cluster[ant_span]
            predicted_clusters[cluster_id].append(current_span)
            span_to_cluster[current_span] = cluster_id

    return predicted_clusters


class XLMRCorefModel(nn.Module):
    def __init__(self,config, hidden_size=768, gcn_hidden_size=768, num_labels=50,training=True):
        super(XLMRCorefModel, self).__init__()
        # Load the pre-trained XLM-RoBERTa model
        self.xlm_roberta = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.gcn = GCNLayer(hidden_size, gcn_hidden_size, num_labels)

        combine_strategy = config["combine_strategy"]
        self.span_width = config["span_width"]
        self.combine_strategy = combine_strategy
        self.training = training

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

    def extract_gold_clusters(self, span_starts, span_ends, cluster_ids):
        """
        Constructs a map from (start, end) -> cluster_id and then builds clusters.
        Returns: List of sets (each set is a gold cluster of span indices)
        """
        gold_clusters = []
        mention_to_cluster = {}
        for i in range(len(span_starts)):
            clusters = defaultdict(list)
            for idx, (start, end, cluster_id) in enumerate(zip(span_starts[i], span_ends[i], cluster_ids[i])):
                if cluster_id == -1:
                    continue
                clusters[cluster_id.item()].append((start.item(), end.item()))
            gold_clusters.append(list(clusters.values()))
        return gold_clusters

    def forward(self, input_ids, attention_mask=None,edges=None,
                span_starts=None, span_ends=None, cluster_ids=None):
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

                dummy_scores = torch.zeros(pruned_span_rep.size(0), 1, device=pruned_span_rep.device)
                ant_scores_with_dummy = torch.cat([dummy_scores, ant_scores], dim=1)

                triu_mask = torch.triu(torch.ones_like(ant_scores, device=ant_scores.device), diagonal=0)

                triu_mask_extended = torch.cat([
                    torch.zeros(triu_mask.size(0), 1, device=triu_mask.device),  # don't mask dummy column
                    triu_mask
                ], dim=1)

                masked_scores = ant_scores_with_dummy.masked_fill(triu_mask_extended.bool(), -1e10)

                # Apply regular softmax
                softmaxed_scores = torch.nn.functional.softmax(masked_scores, dim=-1)

                # Take the log separately, avoiding log(0) issues
                ant_scores = torch.log(softmaxed_scores + 1e-10)
                antecedent_scores_list.append(ant_scores)

        if self.training:
            # only if is training I compute the loss
            gold_clusters = self.extract_gold_clusters(span_starts, span_ends, cluster_ids)
            gold_antecedents_list = []
            for i, pruned_idxs in enumerate(pruned_indices):
                candidate_spans = all_spans[i]
                spans = [candidate_spans[idx] for idx in pruned_idxs]
                span_to_index = {span: j for j, span in enumerate(spans)}

                gold_antecedents = []
                cluster = gold_clusters[i]
                span_cluster_map = {}
                for cluster_spans in cluster:
                    for span in cluster_spans:
                        span_cluster_map[span] = cluster_spans

                for j, span in enumerate(spans):
                    ant_indices = []
                    if span in span_cluster_map:
                        for ant in span_cluster_map[span]:
                            if ant == span:
                                break  # Only consider previous mentions
                            if ant in span_to_index:
                                ant_indices.append(span_to_index[ant])
                    gold_antecedents.append(ant_indices)
                gold_antecedents_list.append(gold_antecedents)

            coref_losses = []
            for i, ant_scores in enumerate(antecedent_scores_list):
                coref_loss = self.compute_coref_loss(ant_scores, gold_antecedents_list[i])
                coref_losses.append(coref_loss)

            avg_loss = torch.stack(coref_losses).mean()

            return avg_loss
        else:
            return antecedent_scores_list, pruned_indices, all_spans

    def compute_coref_loss(self, antecedent_scores, gold_antecedents):
        loss = 0.0
        for i in range(len(antecedent_scores)):
            gold_ants = gold_antecedents[i]
            if not gold_ants:
                # If no gold antecedents, assume the dummy is correct (index 0)
                gold_scores = antecedent_scores[i, 0:1]  # Select dummy antecedent score
            else:
                gold_scores = torch.stack([antecedent_scores[i, j] for j in gold_ants])
            log_sum_exp = torch.logsumexp(antecedent_scores[i], dim=0)
            gold_log_sum_exp = torch.logsumexp(gold_scores, dim=0)
            loss += log_sum_exp - gold_log_sum_exp
        return loss / max(len(antecedent_scores), 1)  # Normalize by number of spans

    @torch.no_grad()
    def predict(self, input_ids, attention_mask, edges):
        self.eval()
        with torch.no_grad():
            antecedent_scores_list, pruned_indices_list, all_spans_list = self.forward(input_ids=input_ids, attention_mask=attention_mask, edges=edges)
        for antecedent_scores, pruned_indices, all_spans in zip(antecedent_scores_list, pruned_indices_list,
                                                                all_spans_list):
            pruned_spans = [all_spans[i] for i in pruned_indices]
            clusters = decode_clusters(pruned_spans, antecedent_scores)
        return clusters


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

