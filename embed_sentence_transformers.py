from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm
import sys

#model_name = 'bert-base-nli-mean-tokens'
model_name = sys.argv[1]
print(model_name)
chunk_size = 8
model = SentenceTransformer(model_name)
model._first_module().max_seq_length = 128

with open("abstracts_specter.json", "r", encoding="utf-8") as f:
    docs = json.load(f)


docs_list = list(docs.values())
print("docs: " + str(len(docs_list)))
out_file_name = "abstracts_embeddings_" + model_name + ".jsonl"
out_file_name = out_file_name.replace(r'/', '_')
print(out_file_name)
with open(out_file_name, "w", encoding="utf-8") as out:
    for i in tqdm(range(0, len(docs_list), chunk_size)):
        chunk = docs_list[i: i + chunk_size]
        abstracts = [c["abstract"] for c in chunk]
        sentence_embeddings = model.encode(abstracts)
        for j in range(len(chunk)):
            out.write(json.dumps({"paper_id": chunk[j]["paper_id"], "embedding": sentence_embeddings[j]
            .tolist()}))
            out.write("\n")
        
        