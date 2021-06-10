import json
import tqdm
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# Build doc to domain mapping
journal_domain_mapping = {
    'Agricultural and Biological Sciences': 'Agr',
    'Biochemistry, Genetics and Molecular Biology': 'Bio',
    'Chemistry': 'Che',
    'Computer Science': 'CS',
    'Earth and Planetary Sciences': "ES",
    'Engineering': 'Eng',
    'Materials Science': 'MS',
    'Mathematics': 'Mat',
    'Medicine and Dentistry': 'Med',
    'Physics and Astronomy': 'Ast'
}


def map_to_domain(journal_domains):
    for jd in journal_domains:
        if journal_domain_mapping.get(jd) is not None:
            return journal_domain_mapping.get(jd)
    return None


def normalize_text(p_text):
    p_text = p_text.replace('\n', ' ').replace('\r', ' ')
    p_text = re.sub(' +', ' ', p_text)
    p_text = p_text.replace(' .', '.').replace(' ,', ',')
    return p_text.strip()


class CitationKG:
    def __init__(self, kg_path, concepts=None, path_document_citations="data/documents_citations.jsonl", path_domain_mapping="data/ccby-domain-mapping.csv"):
        self.kg_path = kg_path
        self.path_document_citations = path_document_citations
        self.path_domain_mapping = path_domain_mapping
        self.doc_to_domain = dict()

        self.entities = dict()
        self.doc_to_entity = dict()
        self.doc_to_mentions = dict()
        self.concepts = concepts
        self.citation_kg = dict()

    def get_entities_for_doc(self, pii):
        entities = self.doc_to_entity.get(pii)
        if entities is None:
            return set()

        if self.concepts is None:
            return set(entities)
        else:
            result = set()
            for e in entities:
                if self.entities[e]["top_most_label"] in self.concepts:
                    result.add(e)
            return result

    def get_all_docs(self):
        return list(self.doc_to_entity.keys())

    def build_kg(self):
        self.read_domain_mapping()
        self.read_entities()
        self.build_citation_graph()
        return self.citation_kg

    def read_entities(self):

        print("reading entities...")
        entity_id = -1
        with open(self.kg_path, "r", encoding="utf8") as f:
            for l in tqdm.tqdm(f):
                entity_id += 1
                e = json.loads(l)
                self.entities[entity_id] = e
                for m in e["mentions"]:
                    pii = m["filename"].split(".")[0]
                    if pii not in self.doc_to_entity:
                        self.doc_to_entity[pii] = list()
                        self.doc_to_mentions[pii] = list()
                    self.doc_to_entity[pii].append(entity_id)
                    self.doc_to_mentions[pii].append(m["text"])

    def read_domain_mapping(self):
        with open(self.path_domain_mapping, mode="r", encoding="utf-8") as f:
            for l in f:
                id, _, domains = l.split("\t")
                journal_domains = [d.strip() for d in domains.split("|")]
                domain = map_to_domain(journal_domains)
                # retain only documents having entities
                self.doc_to_domain[id] = domain

    def build_citation_graph(self):

        print("building citation graph ...")
        title_to_doc = dict()
        redundant_titles = list()

        # Build title to document mapping
        with open(self.path_document_citations, "r", encoding="utf-8") as f:
            for l in tqdm.tqdm(f):
                doc = json.loads(l)
                if doc["pii"] not in self.doc_to_entity:
                    # skip documents without concepts
                    continue

                # collect documents with redundant titles
                title = normalize_text(doc["title"])
                if title in title_to_doc:
                    redundant_titles.append(doc)
                    redundant_titles.append(title_to_doc[title])
                    continue

                title_to_doc[title] = doc
                self.citation_kg[doc["pii"]] = doc
        print("Redundant titles: " + str(len(redundant_titles)))

        # remove documents with redundant titles
        for doc in redundant_titles:
            self.citation_kg.pop(doc["pii"], None)
            title_to_doc.pop(normalize_text(doc["title"]), None)

        # resolve referenced titles to pii
        for pii, doc in self.citation_kg.items():
            doc["domain"] = self.doc_to_domain[doc["pii"]]
            for ref in doc["refs"]:
                ref_title = ref["title"]
                if ref_title in title_to_doc:
                    # in KG citation
                    ref_pii = title_to_doc[ref_title]["pii"]
                    if pii == ref_pii:
                        # do not use self citations
                        continue
                    ref["pii"] = ref_pii

        print(len(self.citation_kg))

    def get_in_kg_refs(self, doc):
        return set([ref["pii"] for ref in doc["refs"] if "pii" in ref])



def get_concept_embeddings(all_docs, citation_kg:CitationKG):
    print("reading concept embeddings...")

    def get_entities_str(pii):
        entities = citation_kg.get_entities_for_doc(pii)
        return ' '.join(str(x) for x in entities)

    def create_tf_idf_vectorizer_entities():
        tfidf_vectorizer = TfidfVectorizer(use_idf=False, token_pattern=r"(?u)\b\w+\b")
        docs_with_entities = [get_entities_str(pii) for pii in citation_kg.doc_to_entity.keys()]
        tfidf_vectorizer.fit(tqdm.tqdm(docs_with_entities))
        return tfidf_vectorizer

    def get_tf_idf_vectors_entities(vectorizer, doc_ids):
        docs_with_entities = [get_entities_str(pii) for pii in doc_ids]
        vectors = vectorizer.transform(docs_with_entities)
        return vectors

    vectorizer_entities = create_tf_idf_vectorizer_entities()
    doc_entities_embeddings = get_tf_idf_vectors_entities(vectorizer_entities, all_docs)
    return normalize(doc_entities_embeddings, "l2", copy=False)

