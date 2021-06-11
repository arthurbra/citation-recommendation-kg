# Citation Recommendation for Research Papers via Knowledge Graphs
This repository contains the source code for the paper:

Brack A., Hoppe A., Ewerth R. (2021) Citation Recommendation for Research Papers via Knowledge Graphs, TPDL 2021 (accepted for publication).
Preprint: https://arxiv.org/abs/2106.05633

# Installation
Python 3.6 required.  Install the requirements with:
- pip install requirements.txt

# Datasets
The evaluation KG is based on the STM-KG provided in the following repository: https://github.com/arthurbra/stm-coref
We use the in-domain KG ('data/stm_silver_kg_in_domain_with_corefs.jsonl') and cross-domain KG ('data/stm_silver_kg_cross_domain_with_corefs.jsonl') for evaluation. 

The file 'data/documents_citations.jsonl' contains the citations and the file 'data/abstracts_specter.json' contains the abstracts for each paper in the STM-KG.

# Embeddings
The file 'data/abstracts_specter_embeddings.json' contains the SPECTER embeddings for each abstract. 
The embeddings can be recreated with the SPECTER library (see also 'embed_specter.py'): https://github.com/allenai/specter

The file 'data/abstracts_embeddings_average_word_embeddings_glove.840B.300d.jsonl' contains GloVe embeddings and the file 'data/abstracts_embeddings_allenai_scibert_scivocab_uncased.jsonl' SciBERT embeddings (averaged).
These embeddings can recreated with the sentence transformers library (see also 'embed_sentence_transformers.py'): https://github.com/UKPLab/sentence-transformers  

# Ranking
To build the citation graph and perform the ranking, run the script 'evaluate_citation_recommendation_ranking.py'.
Check the TODO statements to vary the evaluation (e.g. to change the KG).
