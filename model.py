#This will include:
from collections import defaultdict
from bisect import bisect_left

#XLM-R for contextual embeddings

#GCN over syntactic edges (from UD)

#Span representation module (e.g., endpoint + attention)

#Mention scorer and clusterer

#If you’re replicating SpanBERT-style architecture, you’ll likely use mention-pair scoring or span clustering.

# model.py

import torch
from torch import nn
import  torch.nn.functional as F
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

        degree = torch.zeros(batch_size * seq_len, device=device)
        degree.index_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float))
        degree = degree.clamp(min=1).view(batch_size, seq_len, 1)

        out = out_flat.view(batch_size, seq_len, hidden_dim)
        out = out / degree

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
        # TODO: incoporate this in future
        self.span_width_emb = nn.Embedding(self.span_width, hidden_size)

    def generate_candidate_spans(self, embeddings, attention_mask):
        batch_size, seq_len, _ = embeddings.size()

        all_spans = []
        for b in range(batch_size):


            # Use attention_mask to get actual sequence length
            if attention_mask is not None:
                actual_len = attention_mask[b].sum().item()
            else:
                actual_len = seq_len

            for i in range(actual_len):
                for j in range(i + 1, min(i + self.span_width, actual_len)):
                    all_spans.append((i, j))

        return all_spans

    def prune_spans(self, span_reps, mention_scores, topk_per_doc):
        """
        Args:
            span_reps: (num_spans, span_dim) tensor (since you're using torch.stack now)
            mention_scores: (num_spans,) tensor
            topk_per_doc: scalar tensor or int with topk value
        """
        # Handle the case where topk_per_doc might be a scalar tensor or int
        if isinstance(topk_per_doc, torch.Tensor):
            if topk_per_doc.dim() == 0:  # scalar tensor
                k = min(topk_per_doc.item(), mention_scores.size(0))
            else:  # tensor with dimensions
                k = min(topk_per_doc[0].item(), mention_scores.size(0))
        else:  # regular int/float
            k = min(topk_per_doc, mention_scores.size(0))

        topk_scores, topk_indices = torch.topk(mention_scores, k)
        pruned_span_reps = span_reps[topk_indices]
        pruned_scores = topk_scores
        pruned_indices = topk_indices

        return pruned_span_reps, pruned_scores, pruned_indices

    def get_antecedent_scores(self, span_reps):
        """Score all pairs of spans"""
        if span_reps.dim() == 1:
            span_reps = span_reps.unsqueeze(0)

        num_spans = span_reps.size(0)

        if num_spans == 1:
            # Special case: single span
            pair = torch.cat([span_reps, span_reps], dim=-1)
            return self.antecedent_scorer(pair).view(1, 1)

        # Multiple spans case
        cated = torch.cat([span_reps.unsqueeze(1).expand(-1, num_spans, -1),
                           span_reps.unsqueeze(0).expand(num_spans, -1, -1)], dim=-1)

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

    def create_sentence_mask(self, pruned_idx, spans, sentence_map, sentence_starts, max_sentence_distance=3):
        """
        Create mask to constrain antecedent search based on sentence distance.

        Args:
            pruned_idx: indices of pruned spans
            spans: all candidate spans for this document
            sentence_map: tensor mapping each token to sentence ID
            sentence_starts: tensor of sentence start positions
            max_sentence_distance: maximum allowed sentence distance for antecedents

        Returns:
            mask: boolean tensor (num_spans, num_spans+1) where True = mask out
        """



        num_spans = len(pruned_idx)
        # +1 for dummy antecedent column
        mask = torch.zeros(num_spans, num_spans + 1, dtype=torch.bool, device=sentence_map.device)

        # Get sentence IDs for each pruned span
        span_sentences = []
        for idx in pruned_idx:
            span_start, span_end = spans[idx]
            # Use sentence of the span's head token (or start token)
            span_sent_id = sentence_map[span_start].item()
            span_sentences.append(span_sent_id)

        # Create sentence distance mask
        for i in range(num_spans):  # for each mention
            mention_sent = span_sentences[i]
            for j in range(num_spans):  # for each potential antecedent
                antecedent_sent = span_sentences[j]

                # Mask if antecedent is too far back in sentences
                if mention_sent - antecedent_sent > max_sentence_distance:
                    mask[i, j + 1] = True  # +1 because of dummy column

        return mask

    def forward(self, input_ids, attention_mask=None,edges=None,
                span_starts=None, span_ends=None, cluster_ids=None,
                sentence_map=None, sentence_starts=None):
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

            all_spans = self.generate_candidate_spans(embeddings, attention_mask)

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

            ant_scores = self.get_antecedent_scores(pruned_span_reps)  # (topk, topk)

            dummy_scores = torch.zeros(pruned_span_reps.size(0), 1, device=pruned_span_reps.device)
            ant_scores_with_dummy = torch.cat([dummy_scores, ant_scores], dim=1)

            triu_mask = torch.triu(torch.ones_like(ant_scores, device=ant_scores.device), diagonal=0)

            triu_mask_extended = torch.cat([
                    torch.zeros(triu_mask.size(0), 1, device=triu_mask.device),  # don't mask dummy column
                    triu_mask
                ], dim=1)

            # Create sentence - based mask
            sentence_map = sentence_map.squeeze(0)
            sentence_mask = self.create_sentence_mask(
                    pruned_indices, all_spans, sentence_map, sentence_starts
            )

            # Combine both masks: triangular + sentence constraints
            combined_mask = triu_mask_extended.bool() | sentence_mask

            masked_scores = ant_scores_with_dummy.masked_fill(combined_mask, -1e10)

            # Apply regular softmax
            softmaxed_scores = torch.nn.functional.softmax(masked_scores, dim=-1)

            # Take the log separately, avoiding log(0) issues
            ant_scores = torch.log(softmaxed_scores + 1e-10)


        if self.training:
            mention_loss = self.compute_mention_loss(mention_scores, span_starts, span_ends, cluster_ids, all_spans)

            # it is not predicting any relation, at least none in the first epoch
            #relation_loss = self.compute_coref_training_loss(
            #    span_starts,
            #    span_ends,
            #    cluster_ids,
            #    pruned_indices,
            #    all_spans,
            #    antecedent_scores_list
            #)
            relation_loss = 0
            print(f" Mention loss: {mention_loss} Relation loss: {relation_loss}")
            return relation_loss + mention_loss
        else:
            return ant_scores, pruned_indices, pruned_scores, all_spans

    def compute_coref_training_loss(self, span_starts, span_ends, cluster_ids, pruned_indices, all_spans, antecedent_scores_list):

        gold_clusters = self.extract_gold_clusters(span_starts, span_ends, cluster_ids)[0]

        spans = [all_spans[idx] for idx in pruned_indices]
        span_to_index = {span: j for j, span in enumerate(spans)}

        span_cluster_map = {}
        for cluster_spans in gold_clusters:
            for span in cluster_spans:
                span_cluster_map[span] = cluster_spans

        gold_antecedents = []
        for j, span in enumerate(spans):
            ant_indices = []
            if span in span_cluster_map:
                for ant in span_cluster_map[span]:
                    if ant == span:
                        break  # Only consider previous mentions
                    if ant in span_to_index:
                        ant_indices.append(span_to_index[ant])
            gold_antecedents.append(ant_indices)

        # Compute and return loss
        coref_loss = self.compute_coref_loss(antecedent_scores_list, gold_antecedents)
        return coref_loss

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

    def create_mention_labels(self, num_spans, span_starts, span_ends, cluster_ids, all_spans):
        """
        Create binary labels for mention detection.

        Args:
            num_spans: total number of candidate spans
            span_starts: tensor of gold mention start positions
            span_ends: tensor of gold mention end positions
            cluster_ids: tensor of cluster IDs for gold mentions
            all_spans: list of all candidate spans [(start, end), ...]

        Returns:
            labels: tensor of shape (num_spans,) with 1 for mentions, 0 for non-mentions
        """


        labels = torch.zeros(num_spans, dtype=torch.float)

        # Create set of gold mention spans for fast lookup
        gold_spans = set()
        for start, end, cluster_id in zip(span_starts, span_ends, cluster_ids):
            if cluster_id != -1:  # -1 typically means not a mention
                gold_spans.add((start.item(), end.item()))

        # Mark candidate spans that match gold mentions
        for i, (start, end) in enumerate(all_spans):
            if (start, end) in gold_spans:
                labels[i] = 1.0

        return labels

    def get_positive_indices(self, span_starts, span_ends, cluster_ids, all_spans):
        """More efficient version using tensor operations"""
        positive_indices = []

        # Filter out spans not in any cluster
        valid_mask = cluster_ids != -1
        valid_starts = span_starts[valid_mask]
        valid_ends = span_ends[valid_mask]

        # Create set of gold spans
        gold_spans = set()
        for start, end in zip(valid_starts, valid_ends):
            gold_spans.add((start.item(), end.item()))

        # Find matching candidate spans
        for i, (start, end) in enumerate(all_spans):
            if (start, end) in gold_spans:
                positive_indices.append(i)

        return torch.tensor(positive_indices, device=span_starts.device)

    def hard_negative_mining(self, mention_scores, positive_indices, negative_indices, ratio=2):
        """
        Select hard negatives - spans that look like mentions but aren't
        """
        # Get scores for all negative spans
        neg_scores = mention_scores[negative_indices]

        # Sort by score (highest scoring negatives are "hardest")
        sorted_neg_indices = negative_indices[torch.argsort(neg_scores, descending=True)]

        # Take top-k hardest negatives
        num_hard_negatives = min(len(sorted_neg_indices), len(positive_indices) * ratio)
        hard_negatives = sorted_neg_indices[:num_hard_negatives]

        return hard_negatives

    def compute_mention_loss(self, mention_scores_list, span_starts, span_ends, cluster_ids, all_spans):
        """Compute loss for mention detection"""

        positive_indices = self.get_positive_indices(span_starts, span_ends, cluster_ids, all_spans)
        all_indices = torch.arange(len(all_spans), device=mention_scores_list.device)
        negative_indices = all_indices[~torch.isin(all_indices, positive_indices)]

        sampled_indices = self.hard_negative_mining(
            mention_scores_list, positive_indices, negative_indices, ratio=2
        )

        sampled_labels = torch.zeros(len(sampled_indices), device=mention_scores_list.device)
        sampled_labels[:len(positive_indices)] = 1.0  # First len(positive_indices) are positive

        sampled_scores = mention_scores_list[sampled_indices]
        loss = F.binary_cross_entropy_with_logits(sampled_scores, sampled_labels)

        return  loss

    def inference(self, input_ids, attention_mask=None, edges=None,
                  sentence_map=None, sentence_starts=None, threshold=0.5):
        """
        Perform inference to get coreference clusters

        Args:
            input_ids: Tensor of tokenized input
            attention_mask: Tensor of attention mask
            edges: list of edge lists for GCN
            sentence_map: mapping from tokens to sentences
            sentence_starts: sentence start positions
            threshold: threshold for mention selection (default: 0.5)

        Returns:
            clusters: List of clusters, where each cluster is a list of spans
            mentions: List of all detected mentions
        """
        # Set model to evaluation mode
        self.eval()

        with torch.no_grad():
            # Get model outputs
            antecedent_scores_list, pruned_indices, mention_scores, all_spans = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                edges=edges,
                sentence_map=sentence_map,
                sentence_starts=sentence_starts
            )


            clusters, mentions, scores = self._decode_clusters(
                  antecedent_scores_list,
                  pruned_indices,
                  all_spans,
                  mention_scores,
                  mention_threshold=threshold
            )

            return clusters, mentions, scores

    def _decode_clusters(self, antecedent_scores, pruned_indices, all_spans,
                         mention_scores=None, mention_threshold=0.5):
        """
        Decode antecedent scores into coreference clusters with mention filtering

        Args:
            antecedent_scores: (num_spans, num_spans + 1) antecedent scores
            pruned_indices: indices of pruned spans
            all_spans: all candidate spans
            mention_scores: actual mention scores from mention_scorer (optional)
            mention_threshold: threshold for mention detection

        Returns:
            clusters: List of clusters
            mentions: List of detected mentions
        """
        # Get spans for pruned indices
        pruned_spans = [all_spans[i] for i in pruned_indices]

        # Filter by mention scores
        if mention_scores is not None:
            # Use actual mention scores
            mention_probs = torch.sigmoid(mention_scores.squeeze(-1))
            mention_mask = mention_probs.squeeze(-1) > mention_threshold
        else:
            # Fallback to deriving from antecedent scores
            antecedent_scores = torch.stack(antecedent_scores)
            probs = torch.exp(antecedent_scores)
            mention_probs = 1 - probs[:, 0]
            mention_mask = mention_probs > mention_threshold

        valid_indices = torch.where(mention_mask)[0]

        if len(valid_indices) == 0:
            return [], [], []

        # Get valid spans
        valid_spans = [pruned_spans[i] for i in valid_indices]
        valid_scores = mention_probs[valid_indices].cpu().numpy().tolist()

        # Filter antecedent scores to only include valid mentions
        filtered_antecedent_scores = antecedent_scores[valid_indices]

        # Build clusters using filtered spans
        clusters = self._build_clusters(
            filtered_antecedent_scores,
            valid_indices,
            valid_spans
        )

        return clusters, valid_spans, valid_scores

    def _build_clusters(self, antecedent_scores, mention_indices, spans):
        """
        Build coreference clusters from antecedent scores

        Args:
            antecedent_scores: scores for valid mentions only
            mention_indices: indices of valid mentions
            spans: corresponding spans

        Returns:
            clusters: List of clusters
        """
        # Create mapping from original indices to filtered indices
        index_map = {orig_idx.item(): new_idx for new_idx, orig_idx in enumerate(mention_indices)}

        # Find best antecedent for each mention
        antecedent_links = {}

        for i, scores in enumerate(antecedent_scores):
            # Skip dummy antecedent (index 0)
            ant_scores = scores[1:]  # Remove dummy column

            if len(ant_scores) == 0:
                continue

            # Only consider antecedents that come before current mention
            valid_antecedents = ant_scores[:i]  # Only previous mentions

            if len(valid_antecedents) == 0:
                continue

            # Find best antecedent
            best_ant_rel_idx = torch.argmax(valid_antecedents).item()
            best_score = valid_antecedents[best_ant_rel_idx].item()

            # Only create link if score is reasonable (not too low)
            if best_score > -5.0:  # Adjust threshold as needed
                antecedent_links[i] = best_ant_rel_idx

        # Build clusters using Union-Find
        clusters = self._union_find_clustering(antecedent_links, len(spans))

        # Convert cluster indices to actual spans
        span_clusters = []
        for cluster in clusters:
            span_cluster = [spans[i] for i in cluster]
            span_clusters.append(span_cluster)

        return span_clusters

    def _union_find_clustering(self, antecedent_links, num_mentions):
        """
        Use Union-Find to build clusters from antecedent links

        Args:
            antecedent_links: dict mapping mention -> antecedent
            num_mentions: total number of mentions

        Returns:
            clusters: List of clusters (each cluster is a list of mention indices)
        """
        # Initialize parent array for Union-Find
        parent = list(range(num_mentions))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Union mentions based on antecedent links
        for mention, antecedent in antecedent_links.items():
            union(mention, antecedent)

        # Group mentions by their root parent
        clusters_dict = defaultdict(list)
        for i in range(num_mentions):
            root = find(i)
            clusters_dict[root].append(i)

        # Convert to list of clusters (filter out singleton clusters if desired)
        clusters = [cluster for cluster in clusters_dict.values() if len(cluster) > 1]

        return clusters

    def predict_batch(self, batch_data, tokenizer, threshold=0.5):
        """
        Convenient method for batch prediction

        Args:
            batch_data: dict containing input_ids, attention_mask, edges, etc.
            tokenizer: the model tokenizer to decode to text
            threshold: mention detection threshold

        Returns:
            predictions: List of predictions for each document
        """
        input_ids = batch_data['input_ids']
        attention_mask = batch_data.get('attention_mask')
        edges = batch_data.get('edges')
        sentence_map = batch_data.get('sentence_map')
        sentence_starts = batch_data.get('sentence_starts')

        clusters_list, mentions_list, mention_scores_list = self.inference(
            input_ids=input_ids,
            attention_mask=attention_mask,
            edges=edges,
            sentence_map=sentence_map,
            sentence_starts=sentence_starts,
            threshold=threshold
        )

        predictions = []
        for idx, cluster in enumerate(clusters_list):
            sorted_spans = sorted(cluster)

            mentions_with_text = []
            for mention_span, score in zip(mentions_list, mention_scores_list):
                idx_span = bisect_left(sorted_spans, mention_span)
                idx_span = idx_span if idx_span < len(sorted_spans) else idx_span - 1

                # this spans do not belong to this cluster
                if sorted_spans[idx_span] != mention_span:
                    continue
                start_idx, end_idx = mention_span
                # Extract tokens and decode to text
                mention_tokens = input_ids[0,start_idx:end_idx + 1]
                mention_text = tokenizer.decode(mention_tokens, skip_special_tokens=True)

                mentions_with_text.append({
                    'span': mention_span,
                    'text': mention_text.strip(),
                    'score': score
                })


            predictions.append({
                'cluster_id': idx,
                'mentions_txt':mentions_with_text
            })

        return predictions


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
        for start, end in all_spans:
            span_emb = embeddings[0, start:end+1]  # (span_len, hidden)
            start_emb = embeddings[0, start]       # (hidden)
            end_emb = embeddings[0, end]           # (hidden)

             # Attention-based pooling over the span
            attn_scores = self.attention(span_emb)            # (span_len, 1)
            attn_weights = torch.softmax(attn_scores, dim=0)  # (span_len, 1)
            attn_output = torch.sum(attn_weights * span_emb, dim=0)  # (hidden)

            span_vec = torch.cat([start_emb, end_emb, attn_output], dim=-1)  # (3 * hidden)

            span_reps.append(span_vec)  # (num_spans, 3 * hidden)

        return torch.stack(span_reps)


class MentionScorer(nn.Module):
    def __init__(self, span_repr_size):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(span_repr_size, span_repr_size),
            nn.ReLU(),
            nn.Linear(span_repr_size, 1)
        )

    def forward(self, span_reps):
        mention_scores = self.scorer(span_reps).squeeze(-1)  # (num_spans,)
        return mention_scores

