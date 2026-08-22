"""
dense_retriever.py
Real dense semantic retriever using PubMedBERT sentence embeddings.
Uses sentence-transformers for medical bi-encoder cosine retrieval.
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

_EMBED_MODEL = None

def load_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda:0"):
    """Loads a sentence-transformer model for dense retrieval."""
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL
    
    print(f"Loading embedding model {model_name} on {device}...")
    _EMBED_MODEL = SentenceTransformer(model_name, device=device)
    print(f"Embedding model loaded: {model_name}")
    return _EMBED_MODEL

class RealDenseRetriever:
    """Dense semantic retriever with real sentence-transformer embeddings."""
    
    def __init__(self, corpus, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda:0"):
        self.corpus = corpus
        self.doc_ids = [d["doc_id"] for d in corpus]
        self.texts = [f"{d['title']}. {d['text']}" for d in corpus]
        
        self.model = load_embedding_model(model_name, device)
        
        # Pre-compute corpus embeddings
        print(f"Encoding {len(self.texts)} corpus documents...")
        self.doc_embeddings = self.model.encode(
            self.texts, 
            convert_to_numpy=True, 
            show_progress_bar=False,
            normalize_embeddings=True
        )
        print(f"Corpus embeddings shape: {self.doc_embeddings.shape}")
    
    def retrieve(self, query_text, top_k=None):
        """Returns {doc_id: cosine_similarity} for all corpus documents."""
        q_emb = self.model.encode(
            [query_text], 
            convert_to_numpy=True, 
            normalize_embeddings=True
        )
        
        # Cosine similarity (embeddings are already L2-normalized)
        sims = (q_emb @ self.doc_embeddings.T)[0]
        
        scores = {doc_id: float(sim) for doc_id, sim in zip(self.doc_ids, sims)}
        return scores
    
    def retrieve_top_k(self, query_text, k=3):
        """Returns top-k (doc_id, score) pairs sorted by similarity."""
        scores = self.retrieve(query_text)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return ranked
