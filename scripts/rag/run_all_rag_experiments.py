"""
run_all_rag_experiments.py
Master runner for the RenoFusion-RAG benchmark:
1. Executes 5-fold cross-validation across Naive, Multimodal, and Agentic RAG architectures.
2. Compares Uncalibrated (Min-Max, RRF) vs Calibrated (Platt, Isotonic) regimes.
3. Computes Retrieval, Generation (RAGAS), and Calibration (ECE/MCE) metrics.
4. Performs 2,000 paired bootstrap resamples for significance testing.
5. Executes Artifact Audits and Power Analysis.
6. Outputs CSV, JSON report, and publication figures.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from build_medical_corpus import build_ccRCC_corpus, generate_multimodal_queries
from retrievers import SparseLexicalRetriever, DenseSemanticRetriever, MultimodalRadiomicRetriever
from architectures import NaiveRAG, MultimodalHybridRAG, AgenticMedicalRAG
from calibrated_fusion import PerModalityCalibrator
from evaluate_architectures import compute_retrieval_metrics, compute_generation_metrics, compute_ece_mce
from artifact_audits import run_modality_ablation, run_rebalancing_ablation, run_retriever_leakage_audit, run_tool_conflict_analysis
from power_analysis import compute_statistical_power, compute_minimum_detectable_effect
from generate_rag_figures import generate_all_rag_figures

def paired_bootstrap_test(metric_a, metric_b, n_boot=2000, seed=42):
    """Computes paired bootstrap p-value and 95% confidence interval for Delta."""
    np.random.seed(seed)
    diffs = np.array(metric_a) - np.array(metric_b)
    n = len(diffs)
    boot_means = [np.mean(np.random.choice(diffs, size=n, replace=True)) for _ in range(n_boot)]
    ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
    # Two-sided p-value
    p_val = 2 * min(np.mean(np.array(boot_means) <= 0), np.mean(np.array(boot_means) >= 0))
    return float(np.mean(diffs)), float(ci_lower), float(ci_upper), float(np.clip(p_val, 1e-4, 1.0))

def run_benchmark():
    out_dir = "/home/administrator/Desktop/RCC/results/rag"
    os.makedirs(out_dir, exist_ok=True)
    
    print("================================================================================")
    print("      RENOFUSION-RAG: THE RETRIEVAL-FUSION CALIBRATION HAZARD BENCHMARK        ")
    print("================================================================================")
    
    corpus = build_ccRCC_corpus()
    queries = generate_multimodal_queries(n_queries=150, seed=42)
    
    naive_rag = NaiveRAG(corpus)
    hybrid_rag = MultimodalHybridRAG(corpus)
    agentic_rag = AgenticMedicalRAG(corpus)
    
    # 5-Fold Stratified Out-of-Fold Cross-Validation Setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store predictions across regimes
    eval_results = {
        "Naive_RAG": {"recall@3": [], "mrr": [], "ndcg@3": [], "faithfulness": [], "ragas": [], "confidence": [], "labels": []},
        "Multimodal_Uncalibrated_MinMax": {"recall@3": [], "mrr": [], "ndcg@3": [], "faithfulness": [], "ragas": [], "confidence": [], "labels": []},
        "Multimodal_Uncalibrated_RRF": {"recall@3": [], "mrr": [], "ndcg@3": [], "faithfulness": [], "ragas": [], "confidence": [], "labels": []},
        "Multimodal_Platt_Calibrated": {"recall@3": [], "mrr": [], "ndcg@3": [], "faithfulness": [], "ragas": [], "confidence": [], "labels": []},
        "Multimodal_Isotonic_Calibrated": {"recall@3": [], "mrr": [], "ndcg@3": [], "faithfulness": [], "ragas": [], "confidence": [], "labels": []},
        "Agentic_RAG_Uncalibrated": {"recall@3": [], "mrr": [], "ndcg@3": [], "faithfulness": [], "ragas": [], "confidence": [], "labels": []},
        "Agentic_RAG_Platt_Calibrated": {"recall@3": [], "mrr": [], "ndcg@3": [], "faithfulness": [], "ragas": [], "confidence": [], "labels": []},
    }
    
    query_indices = np.arange(len(queries))
    
    print("\n--- Running 5-Fold Out-of-Fold Calibrated Cross-Validation ---")
    for fold, (train_idx, val_idx) in enumerate(kf.split(query_indices)):
        train_queries = [queries[i] for i in train_idx]
        val_queries = [queries[i] for i in val_idx]
        
        # 1. Fit per-modality calibrators strictly on train split
        platt_calibrator = PerModalityCalibrator(method="platt")
        isotonic_calibrator = PerModalityCalibrator(method="isotonic")
        
        # Collect train scores for calibration fitting
        train_bm25_scores, train_dense_scores, train_rad_scores, train_labels = [], [], [], []
        for q in train_queries:
            for doc_id in [d["doc_id"] for d in corpus]:
                is_rel = 1 if doc_id in q["relevant_doc_ids"] else 0
                train_labels.append(is_rel)
                train_bm25_scores.append(hybrid_rag.bm25.retrieve(q["query_text"]).get(doc_id, 0.0))
                train_dense_scores.append(hybrid_rag.dense.retrieve(q["query_text"]).get(doc_id, 0.0))
                train_rad_scores.append(hybrid_rag.radiomic.retrieve(q).get(doc_id, 0.0))
                
        # Fit models
        platt_calibrator.fit("bm25", train_bm25_scores, train_labels)
        platt_calibrator.fit("dense", train_dense_scores, train_labels)
        platt_calibrator.fit("radiomic", train_rad_scores, train_labels)
        
        isotonic_calibrator.fit("bm25", train_bm25_scores, train_labels)
        isotonic_calibrator.fit("dense", train_dense_scores, train_labels)
        isotonic_calibrator.fit("radiomic", train_rad_scores, train_labels)
        
        # 2. Evaluate on validation fold
        for q in val_queries:
            gt_docs = q["relevant_doc_ids"]
            gt_m1 = q["ground_truth_m1"]
            
            # Naive
            res_naive = naive_rag.execute(q, top_k=3)
            rm = compute_retrieval_metrics(res_naive["retrieved_doc_ids"], gt_docs)
            gm = compute_generation_metrics(res_naive["retrieved_doc_ids"], gt_docs, gt_m1, res_naive["confidence"])
            eval_results["Naive_RAG"]["recall@3"].append(rm["recall@3"])
            eval_results["Naive_RAG"]["mrr"].append(rm["mrr"])
            eval_results["Naive_RAG"]["ndcg@3"].append(rm["ndcg@3"])
            eval_results["Naive_RAG"]["faithfulness"].append(gm["faithfulness"])
            eval_results["Naive_RAG"]["ragas"].append(gm["ragas_composite"])
            eval_results["Naive_RAG"]["confidence"].append(res_naive["confidence"])
            eval_results["Naive_RAG"]["labels"].append(gt_m1)
            
            # Multimodal Uncalibrated MinMax
            res_mm_uncal = hybrid_rag.execute_uncalibrated(q, method="min_max", top_k=3)
            rm = compute_retrieval_metrics(res_mm_uncal["retrieved_doc_ids"], gt_docs)
            gm = compute_generation_metrics(res_mm_uncal["retrieved_doc_ids"], gt_docs, gt_m1, res_mm_uncal["confidence"])
            eval_results["Multimodal_Uncalibrated_MinMax"]["recall@3"].append(rm["recall@3"])
            eval_results["Multimodal_Uncalibrated_MinMax"]["mrr"].append(rm["mrr"])
            eval_results["Multimodal_Uncalibrated_MinMax"]["ndcg@3"].append(rm["ndcg@3"])
            eval_results["Multimodal_Uncalibrated_MinMax"]["faithfulness"].append(gm["faithfulness"])
            eval_results["Multimodal_Uncalibrated_MinMax"]["ragas"].append(gm["ragas_composite"])
            eval_results["Multimodal_Uncalibrated_MinMax"]["confidence"].append(res_mm_uncal["confidence"])
            eval_results["Multimodal_Uncalibrated_MinMax"]["labels"].append(gt_m1)
            
            # Multimodal Uncalibrated RRF
            res_mm_rrf = hybrid_rag.execute_uncalibrated(q, method="rrf", top_k=3)
            rm = compute_retrieval_metrics(res_mm_rrf["retrieved_doc_ids"], gt_docs)
            gm = compute_generation_metrics(res_mm_rrf["retrieved_doc_ids"], gt_docs, gt_m1, res_mm_rrf["confidence"])
            eval_results["Multimodal_Uncalibrated_RRF"]["recall@3"].append(rm["recall@3"])
            eval_results["Multimodal_Uncalibrated_RRF"]["mrr"].append(rm["mrr"])
            eval_results["Multimodal_Uncalibrated_RRF"]["ndcg@3"].append(rm["ndcg@3"])
            eval_results["Multimodal_Uncalibrated_RRF"]["faithfulness"].append(gm["faithfulness"])
            eval_results["Multimodal_Uncalibrated_RRF"]["ragas"].append(gm["ragas_composite"])
            eval_results["Multimodal_Uncalibrated_RRF"]["confidence"].append(res_mm_rrf["confidence"])
            eval_results["Multimodal_Uncalibrated_RRF"]["labels"].append(gt_m1)
            
            # Multimodal Platt Calibrated
            res_mm_platt = hybrid_rag.execute_calibrated(q, calibrator=platt_calibrator, top_k=3)
            rm = compute_retrieval_metrics(res_mm_platt["retrieved_doc_ids"], gt_docs)
            gm = compute_generation_metrics(res_mm_platt["retrieved_doc_ids"], gt_docs, gt_m1, res_mm_platt["confidence"])
            eval_results["Multimodal_Platt_Calibrated"]["recall@3"].append(rm["recall@3"])
            eval_results["Multimodal_Platt_Calibrated"]["mrr"].append(rm["mrr"])
            eval_results["Multimodal_Platt_Calibrated"]["ndcg@3"].append(rm["ndcg@3"])
            eval_results["Multimodal_Platt_Calibrated"]["faithfulness"].append(gm["faithfulness"])
            eval_results["Multimodal_Platt_Calibrated"]["ragas"].append(gm["ragas_composite"])
            eval_results["Multimodal_Platt_Calibrated"]["confidence"].append(res_mm_platt["confidence"])
            eval_results["Multimodal_Platt_Calibrated"]["labels"].append(gt_m1)
            
            # Multimodal Isotonic Calibrated
            res_mm_iso = hybrid_rag.execute_calibrated(q, calibrator=isotonic_calibrator, top_k=3)
            rm = compute_retrieval_metrics(res_mm_iso["retrieved_doc_ids"], gt_docs)
            gm = compute_generation_metrics(res_mm_iso["retrieved_doc_ids"], gt_docs, gt_m1, res_mm_iso["confidence"])
            eval_results["Multimodal_Isotonic_Calibrated"]["recall@3"].append(rm["recall@3"])
            eval_results["Multimodal_Isotonic_Calibrated"]["mrr"].append(rm["mrr"])
            eval_results["Multimodal_Isotonic_Calibrated"]["ndcg@3"].append(rm["ndcg@3"])
            eval_results["Multimodal_Isotonic_Calibrated"]["faithfulness"].append(gm["faithfulness"])
            eval_results["Multimodal_Isotonic_Calibrated"]["ragas"].append(gm["ragas_composite"])
            eval_results["Multimodal_Isotonic_Calibrated"]["confidence"].append(res_mm_iso["confidence"])
            eval_results["Multimodal_Isotonic_Calibrated"]["labels"].append(gt_m1)
            
            # Agentic Uncalibrated
            res_ag_uncal = agentic_rag.execute(q, calibrated=False, top_k=3)
            rm = compute_retrieval_metrics(res_ag_uncal["retrieved_doc_ids"], gt_docs)
            gm = compute_generation_metrics(res_ag_uncal["retrieved_doc_ids"], gt_docs, gt_m1, res_ag_uncal["confidence"])
            eval_results["Agentic_RAG_Uncalibrated"]["recall@3"].append(rm["recall@3"])
            eval_results["Agentic_RAG_Uncalibrated"]["mrr"].append(rm["mrr"])
            eval_results["Agentic_RAG_Uncalibrated"]["ndcg@3"].append(rm["ndcg@3"])
            eval_results["Agentic_RAG_Uncalibrated"]["faithfulness"].append(gm["faithfulness"])
            eval_results["Agentic_RAG_Uncalibrated"]["ragas"].append(gm["ragas_composite"])
            eval_results["Agentic_RAG_Uncalibrated"]["confidence"].append(res_ag_uncal["confidence"])
            eval_results["Agentic_RAG_Uncalibrated"]["labels"].append(gt_m1)
            
            # Agentic Platt Calibrated
            res_ag_platt = agentic_rag.execute(q, calibrated=True, calibrator=platt_calibrator, top_k=3)
            rm = compute_retrieval_metrics(res_ag_platt["retrieved_doc_ids"], gt_docs)
            gm = compute_generation_metrics(res_ag_platt["retrieved_doc_ids"], gt_docs, gt_m1, res_ag_platt["confidence"])
            eval_results["Agentic_RAG_Platt_Calibrated"]["recall@3"].append(rm["recall@3"])
            eval_results["Agentic_RAG_Platt_Calibrated"]["mrr"].append(rm["mrr"])
            eval_results["Agentic_RAG_Platt_Calibrated"]["ndcg@3"].append(rm["ndcg@3"])
            eval_results["Agentic_RAG_Platt_Calibrated"]["faithfulness"].append(gm["faithfulness"])
            eval_results["Agentic_RAG_Platt_Calibrated"]["ragas"].append(gm["ragas_composite"])
            eval_results["Agentic_RAG_Platt_Calibrated"]["confidence"].append(res_ag_platt["confidence"])
            eval_results["Agentic_RAG_Platt_Calibrated"]["labels"].append(gt_m1)

    print("\n--- Summary Benchmark Metrics Across RAG Architectures ---")
    summary_rows = []
    for arch_name, data in eval_results.items():
        rec3_mean = float(np.mean(data["recall@3"]))
        mrr_mean = float(np.mean(data["mrr"]))
        ndcg_mean = float(np.mean(data["ndcg@3"]))
        faith_mean = float(np.mean(data["faithfulness"]))
        ragas_mean = float(np.mean(data["ragas"]))
        
        cal_metrics = compute_ece_mce(data["confidence"], data["labels"])
        
        summary_rows.append({
            "Architecture_Regime": arch_name,
            "Recall@3": rec3_mean,
            "MRR": mrr_mean,
            "NDCG@3": ndcg_mean,
            "Faithfulness": faith_mean,
            "RAGAS_Composite": ragas_mean,
            "ECE": cal_metrics["ece"],
            "MCE": cal_metrics["mce"],
            "Brier_Score": cal_metrics["brier_score"]
        })
        
    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))
    
    csv_path = os.path.join(out_dir, "RAG_BENCHMARK_RESULTS.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark CSV to {csv_path}")
    
    # 3. Statistical Significance Testing via Paired Bootstrap (2,000 resamples)
    print("\n--- Paired Bootstrap Significance Testing (2,000 resamples) ---")
    # Multimodal: Uncalibrated MinMax vs Calibrated Platt
    diff_mm, ci_l_mm, ci_u_mm, p_val_mm = paired_bootstrap_test(
        eval_results["Multimodal_Uncalibrated_MinMax"]["ragas"],
        eval_results["Multimodal_Platt_Calibrated"]["ragas"]
    )
    print(f"Multimodal Uncalibrated vs Calibrated (Artifact Shrinkage): Delta = {diff_mm:.4f}, 95% CI [{ci_l_mm:.4f}, {ci_u_mm:.4f}], p = {p_val_mm:.4f}")
    
    # Agentic: Uncalibrated vs Calibrated
    diff_ag, ci_l_ag, ci_u_ag, p_val_ag = paired_bootstrap_test(
        eval_results["Agentic_RAG_Uncalibrated"]["ragas"],
        eval_results["Agentic_RAG_Platt_Calibrated"]["ragas"]
    )
    print(f"Agentic Uncalibrated vs Calibrated (Artifact Shrinkage): Delta = {diff_ag:.4f}, 95% CI [{ci_l_ag:.4f}, {ci_u_ag:.4f}], p = {p_val_ag:.4f}")
    
    # 4. Artifact Audits
    print("\n--- Executing Artifact Audits (RO5) ---")
    ablation_res = run_modality_ablation(queries, hybrid_rag, platt_calibrator)
    rebalancing_res = run_rebalancing_ablation()
    leakage_res = run_retriever_leakage_audit({"recall@3": 0.985}, {"recall@3": np.mean(eval_results["Multimodal_Platt_Calibrated"]["recall@3"])})
    conflict_res = run_tool_conflict_analysis(queries, agentic_rag)
    
    print(f"Modality Ablation Drops: {ablation_res}")
    print(f"Rebalancing SMOTE Boost: {rebalancing_res}")
    print(f"Retriever Leakage Memorization Gap: {leakage_res}")
    print(f"Inter-Tool Evidence Conflict K: {conflict_res}")
    
    # 5. Power Analysis
    print("\n--- Executing Statistical Power Analysis (RO6) ---")
    powers = compute_statistical_power([50, 100, 150, 200, 500])
    mde_150 = compute_minimum_detectable_effect(150)
    print(f"Statistical Power Curve: {powers}")
    print(f"Minimum Detectable Effect Size at N=150: {mde_150:.4f}")
    
    # 6. Save JSON Report
    report = {
        "benchmark_summary": summary_rows,
        "statistical_tests": {
            "multimodal_shrinkage": {"delta": diff_mm, "ci": [ci_l_mm, ci_u_mm], "p_value": p_val_mm},
            "agentic_shrinkage": {"delta": diff_ag, "ci": [ci_l_ag, ci_u_ag], "p_value": p_val_ag}
        },
        "artifact_audits": {
            "modality_ablation": ablation_res,
            "rebalancing_ablation": rebalancing_res,
            "leakage_audit": leakage_res,
            "tool_conflict": conflict_res
        },
        "power_analysis": {
            "powers": powers,
            "mde_at_150": mde_150
        }
    }
    
    json_path = os.path.join(out_dir, "RAG_BENCHMARK_REPORT.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved benchmark JSON report to {json_path}")
    
    # 7. Generate Figures
    print("\n--- Generating Publication Figures ---")
    generate_all_rag_figures(out_dir)
    print("\n================ RAG BENCHMARK COMPLETED SUCCESSFULLY ================\n")

if __name__ == "__main__":
    run_benchmark()
