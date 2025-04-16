# in this file, there the pre-process to apply semantic role labeling models
# Every token/node line contains 10 tab-separated fields (columns).
# The first column is the numeric ID of the token/node, the next column contains the word FORM
# any coreference annotation, if present, will appear in the last column, which is called MISC
import json

# The MISC column is either a single underscore (_), meaning there is no extra annotation, or one
# or more pieces of annotation (typically in the Attribute=Value form), separated by vertical bars (|).

# The annotation pieces relevant for this shared task always start with Entity=; these should be learned
# from the training data and predicted for the test data.

# Any other annotation that is present in the MISC column of the input file should be preserved in the output (especially note that
# if you discard SpaceAfter=No, or introduce a new one, the validator may report the file as invalid).
import udapi
import os
import re
from collections import defaultdict


# Output: One parsed structure per sentence with token-level annotations and dependency arcs.
# 1. Parse UD files using udapi
# Tokens (surface form, lemma, POS tags, etc.)
# Heads and relations for syntactic dependencies
# Sentence boundaries
# 2. Build a JSON structure for the GCN
# 3. Tokenization and Alignment with XLM-R
# 4. Build the GCN Module
#5. Coreference Head

def extract_cluster_ids(coref_cluster: str):
    """
    Parse the Entity=... string from the MISC column and return a list of actions:
    - {'type': 'start', 'cluster': 'e8'}
    - {'type': 'end', 'cluster': 'e9'}
    """

    actions = []

    # Example: (e8-place-1)e9)
    # Match all cluster mentions inside parens or not
    # e.g., ['(e8-place-1)', 'e9)']
    mention_tokens = re.findall(r'\(*e\d+(?:\[[^\]]+\])?(?:-[^()\s]+)*\)*', coref_cluster)

    for token in mention_tokens:
        starts = token.count('(')
        ends = token.count(')')

        # Clean token (remove parens to isolate the cluster ID and attributes)
        token_clean = token.strip('()')
        # Get cluster ID like 'e8', 'e9', etc.
        m = re.match(r'(e\d+)', token_clean)
        if not m:
            continue
        cluster_id = m.group(1)

        # Add "start" events
        for _ in range(starts):
            actions.append({"type": "start", "cluster": cluster_id})
        # Add "end" events
        for _ in range(ends):
            actions.append({"type": "end", "cluster": cluster_id})

    return actions

def detect_lang_from_filename(filename:str):
    return filename.split("_")[0]


def conllu_to_json(f: str, doc: udapi.Document):

    json_graphs = []

    for bundle in doc.bundles:

        tree = bundle.get_tree()

        surface_nodes = list(tree.descendants)
        empty_nodes = list(tree.empty_nodes)
        all_nodes = surface_nodes + empty_nodes

        # Token-level info (only for surface tokens)
        tokens = [node.form for node in surface_nodes]
        pos = [node.upos for node in surface_nodes]

        # Indexing: ord (e.g., "3", "4.1") -> position in `all_nodes`
        ord_to_index = {}
        for idx, node in enumerate(all_nodes):
            ord_to_index[str(node.ord)] = idx

        # Coreference info (only for surface tokens!)
        mentions = []
        open_mentions = defaultdict(list)
        for i, node in enumerate(surface_nodes):
            misc = node.misc
            if misc and 'Entity' in misc:
                actions = extract_cluster_ids(misc['Entity'])
                for action in actions:
                    if action["type"] == "start":
                        open_mentions[action["cluster"]].append(i)
                    elif action["type"] == "end":
                        if open_mentions[action["cluster"]]:
                            start = open_mentions[action["cluster"]].pop()
                            mentions.append({
                                "start": start,
                                "end": i,
                                "cluster": action["cluster"]
                            })

        # Build edge list (using all_nodes and ord_to_index)
        edges = []
        for node in all_nodes:
            if node.parent:
                source = ord_to_index.get(str(node.parent.ord))
                target = ord_to_index.get(str(node.ord))
                if source is not None and target is not None:
                    edges.append({
                        "source": source,
                        "target": target,
                        "label": node.deprel
                    })

        json_graphs.append({
            "tokens": tokens,           # Surface tokens for XLM-R
            "pos": pos,                 # Surface POS tags
            "edges": edges,             # Graph edges over all nodes
            "mentions": mentions,       # Only from surface tokens
            "sent_id": bundle.bundle_id,
            "doc_key": doc.meta["docname"],
            "lang": detect_lang_from_filename(f)
        })

    return json_graphs



def read_data(data_dir:str):
    data_lst = []
    file_lst = os.listdir(data_dir)
    for f in file_lst:
        if f.endswith("hu_korkor-corefud-minidev.conllu"):
            print(f"Reading {f}")
            doc = udapi.Document(os.path.join(data_dir, f))
            data_lst.append((f, doc))
            break
    return data_lst

def main():

    data_dir = "../data/unc-gold-minidev"
    output_dir = "../data/unc-gold-minidev"

    doc_lst = read_data(data_dir)
    for f, doc in doc_lst:
        json_output = conllu_to_json(f, doc)
        json_filename = os.path.join(output_dir, f.replace('.conllu', '.json'))
        with open(json_filename, 'w') as out_f:
            json.dump(json_output, out_f, indent=2, ensure_ascii=False)
        break

if __name__ == "__main__":
    main()
