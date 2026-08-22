"""
run_all_rag_experiments.py (v2)
LLM-powered RenoFusion-RAG master benchmark runner.
Uses Qwen2.5-3B-Instruct for real generation, ReAct agentic loops, and LLM-as-Judge.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.append(os.path.dirname(__file__))

from build_medical_corpus import build_ccRCC_corpus, generate_multimodal_queries
from retrievers import SparseLexicalRetriever, MultimodalRadiomicRetriever
from calibrated_fusion import PerModalityCalibrator
from evaluate_architectures import compute_retrieval_metrics, compute_generation_metrics_with_llm, compute_ece_mce
from power_analysis import compute_statistical_power, compute_minimum_detectable_effect

# Use fewer LLM-judge calls in the first pass (every 5th query) to save time,
# then enable full LLM-judge for statistical tables
USE_LLM_JUDGE_SAMPLING = 3  # Evaluate with LLM-judge every Nth query

def paired_bootstrap_test(metric_a, metric_b, n_boot=2000, seed=42):
    np.random.seed(seed)
    diffs = np.array(metric_a) - np.array(metric_b)
    n = len(diffs)
    boot_means = [float(np.mean(np.random.choice(diffs, size=n, replace=True))) for _ in range(n_boot)]
    ci_lower, ci_upper = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
    p_val = 2 * min(np.mean(np.array(boot_means) <= 0), np.mean(np.array(boot_means) >= 0))
    return float(np.mean(diffs)), ci_lower, ci_upper, float(np.clip(p_val, 1e-4, 1.0))


def run_benchmark():
    out_dir = "/home/administrator/Desktop/RCC/results/rag"
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 80)
    print("   RENOFUSION-RAG v2: LLM-POWERED CALIBRATION HAZARD BENCHMARK")
    print("   Model: Qwen/Qwen2.5-3B-Instruct | GPUs: 4x NVIDIA L40S (48GB)")
    print("=" * 80)
    
    t_start = time.time()
    
    # 1. Build corpus and queries
    print("\n[1/8] Building medical corpus and queries...")
    corpus = build_ccRCC_corpus()
    queries = generate_multimodal_queries(n_queries=150, seed=42)
    print(f"  Corpus: {len(corpus)} documents | Queries: {len(queries)} multimodal ccRCC scenarios")
    
    # 2. Load LLM
    print("\n[2/8] Loading Qwen2.5-3B-Instruct LLM...")
    from llm_engine import load_llm
    load_llm()
    
    # 3. Initialize architectures
    print("\n[3/8] Initializing RAG architectures with real retrievers...")
    from architectures import NaiveRAG, MultimodalHybridRAG, AgenticMedicalRAG
    naive_rag = NaiveRAG(corpus)
    hybrid_rag = MultimodalHybridRAG(corpus)
    agentic_rag = AgenticMedicalRAG(corpus)
    
    # Define evaluation configurations
    CONFIGS = [
        ("Naive_RAG", "naive", None, None),
        ("Multimodal_Uncal_MinMax", "multimodal_uncal", "min_max", None),
        ("Multimodal_Uncal_RRF", "multimodal_uncal", "rrf", None),
        ("Multimodal_Platt_Cal", "multimodal_cal", None, "platt"),
        ("Multimodal_Isotonic_Cal", "multimodal_cal", None, "isotonic"),
        ("Agentic_RAG_Uncal", "agentic", None, None),
        ("Agentic_RAG_Platt_Cal", "agentic_cal", None, "platt"),
    ]
    
    # Initialize results storage
    results = {name: {
        "recall@3": [], "mrr": [], "ndcg@3": [], 
        "faithfulness": [], "relevance": [], "correctness": [],
        "hallucination_rate": [], "ragas": [], "answer_length": [],
        "confidence": [], "labels": [], "answers": []
    } for name, _, _, _ in CONFIGS}
    
    # 4. 5-Fold Cross-Validation
    print("\n[4/8] Running 5-Fold Out-of-Fold Cross-Validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    query_indices = np.arange(len(queries))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(query_indices)):
        print(f"\n  --- Fold {fold+1}/5 (Train: {len(train_idx)}, Val: {len(val_idx)}) ---")
        train_queries = [queries[i] for i in train_idx]
        val_queries = [queries[i] for i in val_idx]
        
        # Fit calibrators on training split
        platt_cal = PerModalityCalibrator(method="platt")
        isotonic_cal = PerModalityCalibrator(method="isotonic")
        
        train_bm25, train_dense, train_rad, train_labels = [], [], [], []
        for q in train_queries:
            for doc_id in [d["doc_id"] for d in corpus]:
                is_rel = 1 if doc_id in q["relevant_doc_ids"] else 0
                train_labels.append(is_rel)
                train_bm25.append(hybrid_rag.bm25.retrieve(q["query_text"]).get(doc_id, 0.0))
                train_dense.append(hybrid_rag.dense.retrieve(q["query_text"]).get(doc_id, 0.0))
                train_rad.append(hybrid_rag.radiomic.retrieve(q).get(doc_id, 0.0))
        
        platt_cal.fit("bm25", train_bm25, train_labels)
        platt_cal.fit("dense", train_dense, train_labels)
        platt_cal.fit("radiomic", train_rad, train_labels)
        isotonic_cal.fit("bm25", train_bm25, train_labels)
        isotonic_cal.fit("dense", train_dense, train_labels)
        isotonic_cal.fit("radiomic", train_rad, train_labels)
        
        # Evaluate each validation query
        for qi, q in enumerate(val_queries):
            gt_docs = q["relevant_doc_ids"]
            gt_m1 = q["ground_truth_m1"]
            use_judge = (qi % USE_LLM_JUDGE_SAMPLING == 0)
            
            for config_name, arch_type, method, cal_method in CONFIGS:
                try:
                    # Execute architecture
                    if arch_type == "naive":
                        res = naive_rag.execute(q, top_k=3)
                    elif arch_type == "multimodal_uncal":
                        res = hybrid_rag.execute_uncalibrated(q, method=method, top_k=3)
                    elif arch_type == "multimodal_cal":
                        cal = platt_cal if cal_method == "platt" else isotonic_cal
                        res = hybrid_rag.execute_calibrated(q, calibrator=cal, top_k=3)
                    elif arch_type == "agentic":
                        res = agentic_rag.execute(q, calibrated=False, top_k=3)
                    elif arch_type == "agentic_cal":
                        res = agentic_rag.execute(q, calibrated=True, calibrator=platt_cal, top_k=3)
                    else:
                        continue
                    
                    # Compute retrieval metrics
                    rm = compute_retrieval_metrics(res["retrieved_doc_ids"], gt_docs)
                    
                    # Compute generation metrics
                    gm = compute_generation_metrics_with_llm(
                        res["retrieved_doc_ids"], gt_docs, gt_m1,
                        res["confidence"], res.get("generated_answer", ""),
                        res.get("context_text", ""),
                        use_llm_judge=use_judge
                    )
                    
                    # Store results
                    results[config_name]["recall@3"].append(rm["recall@3"])
                    results[config_name]["mrr"].append(rm["mrr"])
                    results[config_name]["ndcg@3"].append(rm["ndcg@3"])
                    results[config_name]["faithfulness"].append(gm["faithfulness"])
                    results[config_name]["relevance"].append(gm["relevance"])
                    results[config_name]["correctness"].append(gm["correctness"])
                    results[config_name]["hallucination_rate"].append(gm["hallucination_rate"])
                    results[config_name]["ragas"].append(gm["ragas_composite"])
                    results[config_name]["answer_length"].append(gm["answer_length"])
                    results[config_name]["confidence"].append(res["confidence"])
                    results[config_name]["labels"].append(gt_m1)
                    results[config_name]["answers"].append(res.get("generated_answer", "")[:200])
                    
                except Exception as e:
                    print(f"    Warning: {config_name} query {qi} failed: {e}")
                    continue
            
            if (qi + 1) % 5 == 0:
                elapsed = time.time() - t_start
                print(f"    Fold {fold+1}, Query {qi+1}/{len(val_queries)} done ({elapsed:.0f}s elapsed)")
    
    # 5. Compute summary statistics
    print("\n[5/8] Computing benchmark summary statistics...")
    summary_rows = []
    for config_name, _, _, _ in CONFIGS:
        data = results[config_name]
        if not data["recall@3"]:
            continue
        cal_metrics = compute_ece_mce(data["confidence"], data["labels"])
        
        row = {
            "Architecture_Regime": config_name,
            "Recall@3": f"{np.mean(data['recall@3']):.4f}",
            "MRR": f"{np.mean(data['mrr']):.4f}",
            "NDCG@3": f"{np.mean(data['ndcg@3']):.4f}",
            "Faithfulness": f"{np.mean(data['faithfulness']):.4f}",
            "Relevance": f"{np.mean(data['relevance']):.4f}",
            "Correctness": f"{np.mean(data['correctness']):.4f}",
            "Hallucination_Rate": f"{np.mean(data['hallucination_rate']):.4f}",
            "RAGAS_Composite": f"{np.mean(data['ragas']):.4f}",
            "Avg_Answer_Length": f"{np.mean(data['answer_length']):.0f}",
            "ECE": f"{cal_metrics['ece']:.4f}",
            "MCE": f"{cal_metrics['mce']:.4f}",
            "Brier_Score": f"{cal_metrics['brier_score']:.4f}",
        }
        summary_rows.append(row)
    
    df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 80)
    print("TABLE 1 & 2: RETRIEVAL + GENERATION BENCHMARK RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))
    
    csv_path = os.path.join(out_dir, "RAG_BENCHMARK_RESULTS_v2.csv")
    df.to_csv(csv_path, index=False)
    
    # 6. Statistical significance testing
    print("\n[6/8] Paired Bootstrap Significance Testing (2,000 resamples)...")
    sig_tests = {}
    test_pairs = [
        ("Multimodal_Uncal_MinMax", "Multimodal_Platt_Cal", "MinMax vs Platt (Multimodal)"),
        ("Multimodal_Uncal_RRF", "Multimodal_Platt_Cal", "RRF vs Platt (Multimodal)"),
        ("Agentic_RAG_Uncal", "Agentic_RAG_Platt_Cal", "Uncal vs Platt (Agentic)"),
        ("Naive_RAG", "Multimodal_Platt_Cal", "Naive vs Calibrated Multimodal"),
    ]
    
    for name_a, name_b, label in test_pairs:
        if results[name_a]["ragas"] and results[name_b]["ragas"]:
            n = min(len(results[name_a]["ragas"]), len(results[name_b]["ragas"]))
            d, ci_l, ci_u, p = paired_bootstrap_test(
                results[name_a]["ragas"][:n],
                results[name_b]["ragas"][:n]
            )
            print(f"  {label}: Δ = {d:.4f}, 95% CI [{ci_l:.4f}, {ci_u:.4f}], p = {p:.4f}")
            sig_tests[label] = {"delta": d, "ci": [ci_l, ci_u], "p_value": p}
    
    # 7. Power analysis
    print("\n[7/8] Statistical Power Analysis...")
    n_actual = len(results["Naive_RAG"]["ragas"])
    powers = compute_statistical_power([50, 100, n_actual, 200, 500])
    mde = compute_minimum_detectable_effect(n_actual)
    print(f"  N = {n_actual} queries | Power at d=0.30: {powers.get(n_actual, 'N/A')}")
    print(f"  Minimum Detectable Effect Size: d = {mde:.4f}")
    
    # 8. Save comprehensive report
    print("\n[8/8] Saving results and generating figures...")
    
    # Save example LLM-generated answers
    example_answers = {}
    for config_name, _, _, _ in CONFIGS:
        if results[config_name]["answers"]:
            example_answers[config_name] = results[config_name]["answers"][:3]
    
    report = {
        "benchmark_summary": summary_rows,
        "statistical_tests": sig_tests,
        "power_analysis": {"powers": powers, "mde": mde, "n_queries": n_actual},
        "example_answers": example_answers,
        "runtime_seconds": time.time() - t_start
    }
    
    json_path = os.path.join(out_dir, "RAG_BENCHMARK_REPORT_v2.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    # Generate figures
    from generate_rag_figures import generate_all_rag_figures
    generate_all_rag_figures(out_dir)
    
    total_time = time.time() - t_start
    print(f"\n{'=' * 80}")
    print(f"BENCHMARK COMPLETE | Total runtime: {total_time/60:.1f} minutes")
    print(f"Results: {csv_path}")
    print(f"Report:  {json_path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    run_benchmark()
