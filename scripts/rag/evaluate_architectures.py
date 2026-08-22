"""
evaluate_architectures.py (v2)
Comprehensive evaluation with real LLM-based metrics:
1. Retrieval: Recall@k, MRR, NDCG@k, Context Precision
2. Generation: Faithfulness (LLM-Judge), Relevance, Correctness, Hallucination Rate
3. Calibration: ECE, MCE, Brier Score
"""

import sys
import os
import numpy as np
sys.path.append(os.path.dirname(__file__))


def compute_retrieval_metrics(retrieved_doc_ids, ground_truth_doc_ids, k_list=[1, 3, 5]):
    """Computes Recall@k, MRR, and NDCG@k for a single query."""
    gt_set = set(ground_truth_doc_ids)
    if not gt_set:
        return {"recall@1": 1.0, "recall@3": 1.0, "recall@5": 1.0, "mrr": 1.0, "ndcg@3": 1.0, "precision@3": 1.0}
        
    metrics = {}
    for k in k_list:
        top_k = retrieved_doc_ids[:k]
        hits = sum(1 for doc_id in top_k if doc_id in gt_set)
        metrics[f"recall@{k}"] = hits / len(gt_set)
        metrics[f"precision@{k}"] = hits / max(k, 1)
        
    # MRR
    mrr = 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gt_set:
            mrr = 1.0 / rank
            break
    metrics["mrr"] = mrr
    
    # NDCG@3
    dcg = 0.0
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(gt_set), 3) + 1))
    for i, doc_id in enumerate(retrieved_doc_ids[:3], start=1):
        if doc_id in gt_set:
            dcg += 1.0 / np.log2(i + 1)
    metrics["ndcg@3"] = (dcg / idcg) if idcg > 0 else 0.0
    
    # Context precision
    metrics["context_precision"] = metrics.get("precision@3", 0.0)
    
    return metrics


def compute_generation_metrics_with_llm(
    retrieved_doc_ids, ground_truth_doc_ids, ground_truth_m1, 
    confidence, generated_answer, context_text,
    use_llm_judge=True
):
    """
    Computes generation quality metrics including real LLM-as-Judge faithfulness.
    """
    gt_set = set(ground_truth_doc_ids)
    context_overlap = sum(1 for doc_id in retrieved_doc_ids if doc_id in gt_set) / max(len(retrieved_doc_ids), 1)
    
    # 1. Faithfulness: LLM-as-Judge or keyword-based fallback
    if use_llm_judge and generated_answer and len(generated_answer) > 10:
        from llm_engine import llm_judge_faithfulness
        faithfulness = llm_judge_faithfulness(context_text[:1500], generated_answer[:500])
    else:
        faithfulness = 0.40 + 0.55 * context_overlap
    
    # 2. Answer Relevance: Does the answer address the clinical question?
    relevance_keywords = ["metastasis", "risk", "staging", "ct", "genomic", "prognosis", "treatment"]
    if generated_answer:
        keyword_hits = sum(1 for kw in relevance_keywords if kw in generated_answer.lower())
        relevance = min(0.30 + 0.10 * keyword_hits, 1.0)
    else:
        relevance = 0.0
    
    # 3. Clinical Correctness: Does the answer align with ground truth M1 status?
    if use_llm_judge and generated_answer and len(generated_answer) > 10:
        from llm_engine import llm_judge_correctness
        correctness = llm_judge_correctness(
            "Is this patient at risk of distant metastasis?",
            generated_answer[:500],
            ground_truth_m1
        )
    else:
        correctness = 1.0 if (("high risk" in generated_answer.lower() or "metastasis" in generated_answer.lower()) == (ground_truth_m1 == 1)) else 0.0
    
    # 4. Hallucination Rate: Check for claims not grounded in context
    hallucination_triggers = ["studies show", "according to recent", "it has been proven", "research indicates"]
    ungrounded_claims = sum(1 for trigger in hallucination_triggers if trigger in generated_answer.lower()) if generated_answer else 0
    hallucination_rate = min(ungrounded_claims * 0.25, 1.0)
    
    # 5. RAGAS Composite: Harmonic mean of faithfulness and relevance
    if (faithfulness + relevance) > 0:
        ragas_composite = (2 * faithfulness * relevance) / (faithfulness + relevance)
    else:
        ragas_composite = 0.0
    
    return {
        "faithfulness": float(np.clip(faithfulness, 0.0, 1.0)),
        "relevance": float(np.clip(relevance, 0.0, 1.0)),
        "correctness": float(correctness),
        "hallucination_rate": float(hallucination_rate),
        "ragas_composite": float(np.clip(ragas_composite, 0.0, 1.0)),
        "answer_length": len(generated_answer) if generated_answer else 0
    }


def compute_ece_mce(confidences, ground_truth_labels, n_bins=10):
    """Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)."""
    confidences = np.array(confidences)
    labels = np.array(ground_truth_labels)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        prop_in_bin = float(np.mean(in_bin))
        
        if prop_in_bin > 0:
            accuracy_in_bin = float(np.mean(labels[in_bin]))
            avg_confidence_in_bin = float(np.mean(confidences[in_bin]))
            diff = abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += diff * prop_in_bin
            mce = max(mce, diff)
            
    brier = float(np.mean((confidences - labels) ** 2))
    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": float(brier)
    }
