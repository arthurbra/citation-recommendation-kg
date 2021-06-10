from citation_kg import CitationKG, get_concept_embeddings
import random
import numpy as np
import tqdm
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from metrics_eval.ranking_metrics import average_precision
from scipy.sparse import hstack
import json
from itertools import chain, combinations


def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))


def sample_elements(elements, n):
    return np.random.choice(elements, n, replace=False)


def build_kg_citation_testset(citation_kg:CitationKG):
    random.seed(23483)
    docs = citation_kg.get_all_docs()
    # Build KG citations testset
    kg_citation_testset = list()
    for d in citation_kg.citation_kg.values():
        refs = citation_kg.get_in_kg_refs(d)
        min_citations = 4
        if len(refs) >= min_citations:
            domain = d["domain"]
            real_refs = list(refs)
            fake_refs = []
            kg_citation_testset.append({
                "pii": d["pii"],
                "domain": domain,
                "real_refs": real_refs,
                "fake_refs": fake_refs
            })
    print(len(kg_citation_testset))
    return kg_citation_testset


def compute_cosine_similarities(vec, others):
    return linear_kernel(vec, others).flatten()
    #return cosine_similarity(vec, others).flatten()


def rank_documents(out_name, testset, all_docs, all_docs_vectors, dense_embeddings=False, write_ranking=False, like_specter=like_specter):
    avp10s = []
    avp20s = []
    avp50s = []
    avpAlls = []
    if write_ranking:
        f = open(out_name, "w", encoding="utf-8")
    for d in tqdm.tqdm(testset):
        ranked = []
        index = all_docs.index(d["pii"])
        d_vec = all_docs_vectors[index]
        if like_specter:
            to_retrieve = d["fake_refs"] + d["real_refs"]
            to_retrieve_vec = np.array([all_docs_vectors[all_docs.index(r)] for r in to_retrieve])
            if not dense_embeddings:
                to_retrieve_vec = np.array([np.array(r.todense()).reshape(-1) for r in to_retrieve_vec])
        else:
            to_retrieve = all_docs
            to_retrieve_vec = all_docs_vectors
        if dense_embeddings:
            similarities = compute_cosine_similarities([d_vec], to_retrieve_vec)
        else:
            similarities = compute_cosine_similarities(d_vec, to_retrieve_vec)
        for r, sim in zip(to_retrieve, similarities):
            if r == d["pii"]:
                continue
            ranked.append((r, sim))
        ranked.sort(key=lambda e: e[1], reverse=True)
        avp10s.append(average_precision(np.array(d["real_refs"]), np.array([r[0] for r in ranked]), k=10))
        avp20s.append(average_precision(np.array(d["real_refs"]), np.array([r[0] for r in ranked]), k=20))
        avp50s.append(average_precision(np.array(d["real_refs"]), np.array([r[0] for r in ranked]), k=50))
        avpAlls.append(average_precision(np.array(d["real_refs"]), np.array([r[0] for r in ranked]), k=0))

        if write_ranking:
            f.write(json.dumps({
                "pii": d["pii"],
                "domain": d["domain"],
                "citations": d["real_refs"],
                "ranking_result":  ranked,
                "avp10": avp10s[-1]
            }) + "\n")

    if write_ranking:
        f.close()
    return {"MAP@10": np.mean(avp10s), "MAP@20": np.mean(avp20s), "MAP@50": np.mean(avp50s), "MAP@All": np.mean(avpAlls)}



def read_embeddings(path):
    print("reading embeddings: " + path)
    result = dict()
    with open(path, "r") as f:
        for l in tqdm.tqdm(f):
            try:
                doc = json.loads(l)
                result[doc["paper_id"]] = np.array(doc["embedding"])
            except ValueError:
                continue
    return result


def get_embeddings(embeddings, doc_ids):
    result = list()
    for pii in doc_ids:
        emb = embeddings[pii]
        result.append(emb)
    return normalize(np.array(result), "l2")


def evaluate_glove(kg_citation_testset, all_docs):
    print("evaluating glove...")
    doc_glove_embeddings = get_glove_embeddings(all_docs)
    maps = rank_documents("glove_ranking.jsonl", kg_citation_testset, all_docs, doc_glove_embeddings, dense_embeddings=True)
    print(maps)


def get_glove_embeddings(all_docs):
    glove_embeddings = read_embeddings("data/abstracts_embeddings_average_word_embeddings_glove.840B.300d.jsonl")
    doc_glove_embeddings = get_embeddings(glove_embeddings, all_docs)
    return doc_glove_embeddings


def evaluate_scibert(kg_citation_testset, all_docs):
    print("Evaluating SciBERT...")
    doc_scibert_embeddings = get_scibert_embeddings(all_docs)
    maps = rank_documents("scibert_ranking.jsonl", kg_citation_testset, all_docs, doc_scibert_embeddings, dense_embeddings=True)
    print(maps)


def get_scibert_embeddings(all_docs):
    scibert_embeddings = read_embeddings("data/abstracts_embeddings_allenai_scibert_scivocab_uncased.jsonl")
    doc_scibert_embeddings = get_embeddings(scibert_embeddings, all_docs)
    return doc_scibert_embeddings


def evaluate_specter(kg_citation_testset, all_docs):
    print("Evaluating SPECTER...")
    doc_specter_embeddings = get_specter_embeddings(all_docs)
    maps = rank_documents("specter_ranking.jsonl", kg_citation_testset, all_docs, doc_specter_embeddings, dense_embeddings=True)
    print(maps)


def get_specter_embeddings(all_docs):
    specter_embeddings = read_embeddings("data/abstracts_specter_embeddings.json")
    doc_specter_embeddings = get_embeddings(specter_embeddings, all_docs)
    return doc_specter_embeddings


def get_random_embeddings(all_docs):
    random.seed(23483)
    embeddings = np.random.rand(len(all_docs), 200)
    embeddings = normalize(embeddings, "l2")
    return embeddings


def evaluate_concept_vector(kg_citation_testset, citation_kg:CitationKG, all_docs):
    print("Evaluating concept_vector...")
    concept_embeddings = get_concept_embeddings(all_docs, citation_kg)
    maps = rank_documents("concepts_ranking.jsonl", kg_citation_testset, all_docs, concept_embeddings, dense_embeddings=False)
    print(maps)


def evaluate_specter_and_concept_vector(kg_citation_testset, citation_kg:CitationKG, all_docs):
    print("Evaluating SPECTER + concept_vector...")
    doc_concept_embeddings = get_concept_embeddings(all_docs, citation_kg)
    doc_specter_embeddings = get_specter_embeddings(all_docs)
    combined = hstack([doc_concept_embeddings, doc_specter_embeddings]).tocsr()
    combined = normalize(combined, "l2", copy=False)
    maps = rank_documents("specter_and_concepts_ranking.jsonl", kg_citation_testset, all_docs, combined, dense_embeddings=False)
    print(maps)


def evaluate_glove_and_concept_vector(kg_citation_testset, citation_kg:CitationKG, all_docs):
    print("Evaluating GloVe + concept_vector...")
    doc_concept_embeddings = get_concept_embeddings(all_docs, citation_kg)
    doc_glove_embeddings = get_glove_embeddings(all_docs)
    combined = hstack([doc_concept_embeddings, doc_glove_embeddings]).tocsr()
    combined = normalize(combined, "l2", copy=False)
    maps = rank_documents("glove_and_concepts_ranking.jsonl", kg_citation_testset, all_docs, combined, dense_embeddings=False)
    print(maps)


def evaluate_scibert_and_concept_vector(kg_citation_testset, citation_kg:CitationKG, all_docs):
    print("Evaluating SciBERT + concept_vector...")
    doc_concept_embeddings = get_concept_embeddings(all_docs, citation_kg)
    doc_scibert_embeddings = get_scibert_embeddings(all_docs)
    combined = hstack([doc_concept_embeddings, doc_scibert_embeddings]).tocsr()
    combined = normalize(combined, "l2", copy=False)
    maps = rank_documents("scibert_and_concepts_ranking.jsonl", kg_citation_testset, all_docs, combined, dense_embeddings=False)
    print(maps)

def evaluate_random(kg_citation_testset, all_docs):
    print("Evaluating random...")
    embeddings = get_random_embeddings(all_docs)
    maps = rank_documents("random_ranking.jsonl", kg_citation_testset, all_docs, embeddings, dense_embeddings=True)
    print(maps)


# TODO: Adapt path the KG (in-domain or cross-domain KG)
kg_path = r"data\stm_silver_kg_in_domain_with_corefs.jsonl"
#kg_path = r"data\stm_silver_kg_cross_domain_with_corefs.jsonl"

# TODO: to evaluate only certain concept types, switch the for loop
#for concept_set in powerset(["Process", "Method", "Material", "Data"]):
for concept_set in [["Process", "Method", "Material", "Data"]]:

    concept_set = list(concept_set)
    print(kg_path)
    print(concept_set)
    citation_kg = CitationKG(kg_path, concepts=concept_set)
    citation_kg.build_kg()
    kg_citation_testset = build_kg_citation_testset(citation_kg)
    all_docs = citation_kg.get_all_docs()

    evaluate_glove(kg_citation_testset, all_docs)
    evaluate_scibert(kg_citation_testset, all_docs)
    evaluate_specter(kg_citation_testset, all_docs)

    evaluate_scibert_and_concept_vector(kg_citation_testset, citation_kg, all_docs)

    evaluate_specter_and_concept_vector(kg_citation_testset, citation_kg, all_docs)

    evaluate_concept_vector(kg_citation_testset, citation_kg, all_docs)

    evaluate_glove_and_concept_vector(kg_citation_testset, citation_kg, all_docs)

    evaluate_random(kg_citation_testset, all_docs)

