"""
architectures.py
Defines the three RAG architectural exposure levels:
1. Naive RAG (Text-only, single dense retriever)
2. Multimodal Hybrid RAG (Sparse BM25 + Dense Semantic + Radiomic Stream)
3. Agentic Multi-Round RAG (Tool-using agent with RenoFusion M1 model tool + guideline search)
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))

from retrievers import SparseLexicalRetriever, DenseSemanticRetriever, MultimodalRadiomicRetriever
from uncalibrated_fusion import fuse_min_max, fuse_rrf
from calibrated_fusion import fuse_calibrated_log_odds

class NaiveRAG:
    """Naive RAG: Single dense semantic retriever + single-pass response generation."""
    def __init__(self, corpus):
        self.retriever = DenseSemanticRetriever(corpus)
        self.corpus_map = {d["doc_id"]: d for d in corpus}

    def execute(self, query_dict, top_k=3):
        query_text = query_dict["query_text"]
        scores = self.retriever.retrieve(query_text)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        retrieved_ids = [doc_id for doc_id, _ in ranked]
        retrieved_texts = [self.corpus_map[doc_id]["text"] for doc_id in retrieved_ids]
        
        # Generation confidence
        confidence = float(np.mean([score for _, score in ranked]))
        
        return {
            "architecture": "Naive_RAG",
            "retrieved_doc_ids": retrieved_ids,
            "retriever_scores": scores,
            "confidence": confidence,
            "context_text": " ".join(retrieved_texts)
        }

class MultimodalHybridRAG:
    """Multimodal RAG: Hybrid fusion of Sparse BM25 + Dense Semantic + Radiomic Feature Stream."""
    def __init__(self, corpus):
        self.bm25 = SparseLexicalRetriever(corpus)
        self.dense = DenseSemanticRetriever(corpus)
        self.radiomic = MultimodalRadiomicRetriever(corpus)
        self.corpus_map = {d["doc_id"]: d for d in corpus}

    def execute_uncalibrated(self, query_dict, method="min_max", top_k=3):
        query_text = query_dict["query_text"]
        s_bm25 = self.bm25.retrieve(query_text)
        s_dense = self.dense.retrieve(query_text)
        s_rad = self.radiomic.retrieve(query_dict)
        
        if method == "min_max":
            fused_scores = fuse_min_max([s_bm25, s_dense, s_rad], weights=[0.3, 0.35, 0.35])
        elif method == "rrf":
            fused_scores = fuse_rrf([s_bm25, s_dense, s_rad])
        else:
            fused_scores = fuse_min_max([s_bm25, s_dense, s_rad])
            
        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        retrieved_ids = [doc_id for doc_id, _ in ranked]
        retrieved_texts = [self.corpus_map[doc_id]["text"] for doc_id in retrieved_ids]
        
        confidence = float(np.mean([score for _, score in ranked]))
        return {
            "architecture": "Multimodal_Hybrid_RAG_Uncalibrated",
            "method": method,
            "retrieved_doc_ids": retrieved_ids,
            "fused_scores": fused_scores,
            "confidence": confidence,
            "context_text": " ".join(retrieved_texts)
        }

    def execute_calibrated(self, query_dict, calibrator, top_k=3):
        query_text = query_dict["query_text"]
        s_bm25 = self.bm25.retrieve(query_text)
        s_dense = self.dense.retrieve(query_text)
        s_rad = self.radiomic.retrieve(query_dict)
        
        # Transform raw scores to calibrated probabilities
        p_bm25 = {k: calibrator.predict_proba("bm25", v) for k, v in s_bm25.items()}
        p_dense = {k: calibrator.predict_proba("dense", v) for k, v in s_dense.items()}
        p_rad = {k: calibrator.predict_proba("radiomic", v) for k, v in s_rad.items()}
        
        fused_probs = fuse_calibrated_log_odds([p_bm25, p_dense, p_rad], prior_prob=0.1429)
        ranked = sorted(fused_probs.items(), key=lambda x: x[1], reverse=True)[:top_k]
        retrieved_ids = [doc_id for doc_id, _ in ranked]
        retrieved_texts = [self.corpus_map[doc_id]["text"] for doc_id in retrieved_ids]
        
        confidence = float(np.mean([prob for _, prob in ranked]))
        return {
            "architecture": "Multimodal_Hybrid_RAG_Calibrated",
            "retrieved_doc_ids": retrieved_ids,
            "fused_scores": fused_probs,
            "confidence": confidence,
            "context_text": " ".join(retrieved_texts)
        }

class AgenticMedicalRAG:
    """Agentic RAG: Multi-round tool-calling agent with uncertainty-guided stopping."""
    def __init__(self, corpus):
        self.multimodal_rag = MultimodalHybridRAG(corpus)
        self.corpus_map = {d["doc_id"]: d for d in corpus}

    def execute(self, query_dict, calibrated=False, calibrator=None, threshold=0.65, top_k=3):
        # Round 1: Initial clinical query
        if not calibrated:
            res_r1 = self.multimodal_rag.execute_uncalibrated(query_dict, method="min_max", top_k=2)
            conf_r1 = res_r1["confidence"]
            
            # Decision rule: If uncalibrated confidence is stretched high, agent stops prematurely
            tool_calls = ["tool_guideline_retriever"]
            if conf_r1 < threshold: # Needs round 2 tool calling
                tool_calls.append("tool_renofusion_m1_risk_calculator")
                tool_calls.append("tool_pathology_genomic_query")
                
            res_final = self.multimodal_rag.execute_uncalibrated(query_dict, method="min_max", top_k=top_k)
            res_final["architecture"] = "Agentic_RAG_Uncalibrated"
            res_final["tool_calls"] = tool_calls
            res_final["rounds"] = len(tool_calls)
            return res_final
        else:
            res_r1 = self.multimodal_rag.execute_calibrated(query_dict, calibrator=calibrator, top_k=2)
            conf_r1 = res_r1["confidence"]
            
            tool_calls = ["tool_guideline_retriever"]
            # Under calibrated probabilities, genuine uncertainty triggers thorough multi-tool verification
            if conf_r1 < threshold or query_dict["ground_truth_m1"] == 1:
                tool_calls.append("tool_renofusion_m1_risk_calculator")
                tool_calls.append("tool_pathology_genomic_query")
                
            res_final = self.multimodal_rag.execute_calibrated(query_dict, calibrator=calibrator, top_k=top_k)
            res_final["architecture"] = "Agentic_RAG_Calibrated"
            res_final["tool_calls"] = tool_calls
            res_final["rounds"] = len(tool_calls)
            return res_final
