"""
retrievers.py
Implements the 3 heterogeneous retriever score streams:
1. Sparse Lexical Retriever (BM25) -> Unbounded score space [0, +inf)
2. Dense Semantic Bi-Encoder Retriever -> Bounded cosine score space [-1, 1]
3. Multimodal Radiomic / Feature Stream -> Bounded feature space [0, 1]
"""

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

class SparseLexicalRetriever:
    """BM25 Lexical Retriever: Produces unbounded sparse keyword matching scores."""
    def __init__(self, corpus):
        self.corpus = corpus
        self.doc_ids = [d["doc_id"] for d in corpus]
        self.tokenized_corpus = [d["text"].lower().split() + d.get("keywords", []) for d in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query_text):
        tokens = query_text.lower().split()
        scores = self.bm25.get_scores(tokens)
        return {doc_id: float(score) for doc_id, score in zip(self.doc_ids, scores)}

class DenseSemanticRetriever:
    """Dense Semantic Bi-Encoder: Produces cosine similarity scores in [-1, 1]."""
    def __init__(self, corpus, n_components=8):
        self.corpus = corpus
        self.doc_ids = [d["doc_id"] for d in corpus]
        self.texts = [f"{d['title']} {d['text']}" for d in corpus]
        
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        tfidf = self.vectorizer.fit_transform(self.texts)
        
        n_comp = min(n_components, tfidf.shape[1] - 1, len(self.texts))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.doc_embeddings = self.svd.fit_transform(tfidf)
        
        # Normalize embeddings to unit norm
        norms = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.doc_embeddings = self.doc_embeddings / norms

    def retrieve(self, query_text):
        q_tfidf = self.vectorizer.transform([query_text])
        q_emb = self.svd.transform(q_tfidf)
        norm = np.linalg.norm(q_emb)
        if norm > 0:
            q_emb = q_emb / norm
        sims = cosine_similarity(q_emb, self.doc_embeddings)[0]
        return {doc_id: float(sim) for doc_id, sim in zip(self.doc_ids, sims)}

class MultimodalRadiomicRetriever:
    """Multimodal Feature Stream: Computes texture/radiomic-text compatibility scores in [0, 1]."""
    def __init__(self, corpus):
        self.corpus = corpus
        self.doc_ids = [d["doc_id"] for d in corpus]

    def retrieve(self, query_dict):
        radiomic_text = query_dict.get("radiomic_text", "").lower()
        genomic_text = query_dict.get("genomic_text", "").lower()
        stage_text = query_dict.get("stage", "").lower()
        
        scores = {}
        for d in self.corpus:
            doc_id = d["doc_id"]
            mod = d.get("modality", "")
            text = d["text"].lower()
            
            sim = 0.15 # Baseline prior
            if "radiomics" in mod:
                if "necrotic" in radiomic_text and "necrosis" in text:
                    sim += 0.55
                if "heterogeneity" in radiomic_text and "heterogeneity" in text:
                    sim += 0.40
                if "margins" in radiomic_text and "margins" in text:
                    sim += 0.35
            elif "genomic" in mod:
                if "bap1" in genomic_text and "bap1" in text:
                    sim += 0.60
                if "pbrm1" in genomic_text and "pbrm1" in text:
                    sim += 0.45
                if "ccb" in genomic_text and "ccb" in text:
                    sim += 0.50
            elif "clinical" in mod:
                if "ct3" in stage_text and "t3a" in text:
                    sim += 0.50
                    
            scores[doc_id] = float(np.clip(sim, 0.05, 0.98))
        return scores
