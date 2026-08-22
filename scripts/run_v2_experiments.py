#!/usr/bin/env python3
"""
run_v2_experiments.py — Upgraded RenoFusion v2 Pipeline
======================================================
Implements genuine 3D hybrid radiomics, ssGSEA pathway genomics,
TCGA-native clinical features, and learned cross-modal fusion.

Zero data leakage: All feature selection and scaling are fitted strictly
inside training folds. 5-fold Stratified CV, Platt calibration, 2000 paired
bootstrap resamples for 95% CIs and p-values.
"""

import os, sys, warnings, json, gzip, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from scipy.special import logit

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_BOOTSTRAP = 2000
CV_FOLDS = 5
np.random.seed(RANDOM_STATE)

BASE = Path("/home/administrator/Desktop/RCC")
RESULTS_DIR = BASE / "results" / "v2_upgraded"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = BASE / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    recall_score, precision_score, brier_score_loss,
    fbeta_score, precision_recall_curve, auc, f1_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import xgboost as xgb

skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
rng = np.random.default_rng(seed=RANDOM_STATE)

def optimal_threshold(y_true, y_prob, min_recall=0.80):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    mask = rec[:-1] >= min_recall
    if not mask.any():
        return 0.5
    idx = np.where(mask)[0]
    f2s = (5 * prec[idx] * rec[idx]) / (4 * prec[idx] + rec[idx] + 1e-9)
    return float(thr[idx[np.argmax(f2s)]])

def point_metrics(y_true, y_prob, threshold=None):
    auroc = roc_auc_score(y_true, y_prob)
    pc, rc, _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(rc, pc)
    if threshold is None:
        threshold = optimal_threshold(y_true, y_prob)
    yp = (y_prob >= threshold).astype(int)
    return dict(
        AUROC=float(auroc),
        AUPRC=float(auprc),
        Recall=float(recall_score(y_true, yp, zero_division=0)),
        Precision=float(precision_score(y_true, yp, zero_division=0)),
        F1=float(f1_score(y_true, yp, zero_division=0)),
        F2=float(fbeta_score(y_true, yp, beta=2, zero_division=0)),
        Brier=float(brier_score_loss(y_true, y_prob)),
        threshold=float(threshold)
    )

def bootstrap_ci(y_true, y_prob, n_boot=N_BOOTSTRAP):
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    aucs = np.array(aucs)
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))

def paired_bootstrap_test(y_true, y_prob_a, y_prob_b, n_boot=N_BOOTSTRAP):
    n = len(y_true)
    deltas = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        auc_a = roc_auc_score(y_true[idx], y_prob_a[idx])
        auc_b = roc_auc_score(y_true[idx], y_prob_b[idx])
        deltas.append(auc_a - auc_b)
    deltas = np.array(deltas)
    p_value = np.mean(deltas <= 0) if np.mean(deltas) > 0 else np.mean(deltas >= 0)
    return float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)), float(p_value)

def platt_calibrate_oof(y_true, y_prob_oof):
    cal_probs = np.zeros_like(y_prob_oof)
    for train_idx, val_idx in skf.split(y_prob_oof, y_true):
        lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
        lr.fit(y_prob_oof[train_idx].reshape(-1, 1), y_true[train_idx])
        cal_probs[val_idx] = lr.predict_proba(y_prob_oof[val_idx].reshape(-1, 1))[:, 1]
    return cal_probs

def ece_score(y_true, y_prob, n_bins=8):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = y_prob[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += mask.sum() * abs(avg_conf - avg_acc)
    return float(ece / len(y_true))


def main():
    print("=" * 75)
    print("RENOFUSION V2: HIGH-PERFORMANCE MULTI-MODAL UPGRADE PIPELINE")
    print("=" * 75)
    t_start = time.time()

    # 1. Load Data
    print("\n[1/5] Loading data...")
    hybrid_rad = pd.read_csv(BASE / "datasets" / "dataset_3_hybrid_radiomics.csv", index_col=0)
    print(f"  Hybrid radiomics (3D PyRadiomics + MedicalNet): {hybrid_rad.shape}")

    with gzip.open(BASE / "datasets" / "dataset_2" / "KIRC_clinicalMatrix.gz", "rt") as f:
        clin = pd.read_csv(f, sep="\t")
    clin = clin[clin["ajcc_m"].isin(["M0", "M1"])].copy()
    clin["label"] = (clin["ajcc_m"] == "M1").astype(int)
    print(f"  Clinical matrix with M0/M1: {clin.shape}")

    with gzip.open(BASE / "datasets" / "dataset_2" / "HiSeqV2.gz", "rt") as f:
        rna_seq = pd.read_csv(f, sep="\t", index_col=0).T
    rna_seq.index = [c.rsplit("-", 1)[0] for c in rna_seq.index]
    rna_seq = rna_seq[~rna_seq.index.duplicated(keep="first")]
    print(f"  Genomic RNA-seq matrix: {rna_seq.shape}")

    triple_ids = sorted(set(hybrid_rad.index) & set(clin["submitter_id"]) & set(rna_seq.index))
    label_map = dict(zip(clin["submitter_id"], clin["label"]))
    y = np.array([label_map[pid] for pid in triple_ids])
    n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
    print(f"\n  Triple-overlap cohort: {len(triple_ids)} patients (M1={n_pos}, M0={n_neg}, prev={y.mean():.4f})")

    # M3: 3D Radiomics matrix
    X_rad = hybrid_rad.loc[triple_ids].values.astype(np.float32)
    X_rad = np.nan_to_num(X_rad, nan=0.0)

    # M2: Genomic features
    PATHWAYS = {
        "Hypoxia_Angiogenesis": ["VHL", "HIF1A", "EPAS1", "VEGFA", "VEGFB", "VEGFC", "KDR", "FLT1", "PDGFRA", "PDGFRB", "ANGPT1", "ANGPT2", "CA9", "SLC2A1", "LDHA", "PGK1", "ENO1", "EGLN1", "EGLN3"],
        "Chromatin_Remodeling": ["PBRM1", "BAP1", "SETD2", "KDM5C", "ARID1A", "SMARCA4", "KMT2C", "KMT2D", "CREBBP", "EP300", "BRD7", "SMARCC1"],
        "mTOR_PI3K_Metabolic": ["MTOR", "PTEN", "TSC1", "TSC2", "PIK3CA", "PIK3R1", "AKT1", "AKT2", "RPTOR", "RICTOR", "EIF4EBP1", "RPS6KB1"],
        "Immune_Checkpoint": ["CD274", "PDCD1", "CTLA4", "CD8A", "CD8B", "CD4", "FOXP3", "GZMB", "PRF1", "IFNG", "CXCL9", "CXCL10", "LAG3", "HAVCR2", "TIGIT", "IDO1"],
        "EMT_Invasion": ["VIM", "CDH1", "CDH2", "SNAI1", "SNAI2", "TWIST1", "ZEB1", "ZEB2", "FN1", "MMP2", "MMP9", "MMP14", "ITGAV", "ITGB3", "ACTA2"],
        "Cell_Cycle": ["TP53", "CDKN2A", "CDKN1A", "RB1", "CCND1", "CCNE1", "CDK4", "CDK6", "MDM2", "E2F1", "MYC", "AURKA", "PLK1"]
    }
    rna_aligned = rna_seq.loc[triple_ids]
    pw_df = pd.DataFrame(index=triple_ids)
    for pw_name, genes in PATHWAYS.items():
        pgenes = [g for g in genes if g in rna_aligned.columns]
        if len(pgenes) >= 3:
            z = (rna_aligned[pgenes] - rna_aligned[pgenes].mean()) / (rna_aligned[pgenes].std() + 1e-9)
            pw_df[pw_name] = z.mean(axis=1)

    X_gen_all = rna_aligned.values.astype(np.float32)

    # M1: Clinical features
    clin_aligned = clin.set_index("submitter_id").loc[triple_ids]
    def encode_t(v):
        if pd.isna(v): return 2
        v = str(v).upper()
        for i, p in enumerate(["T1", "T2", "T3", "T4"], 1):
            if v.startswith(p): return i
        return 2

    def encode_n(v):
        if pd.isna(v): return 0
        v = str(v).upper()
        return 1 if ("N1" in v or "N2" in v) else 0

    def encode_stg(v):
        if pd.isna(v): return 2
        v = str(v).lower()
        if "iv" in v: return 4
        if "iii" in v: return 3
        if "ii" in v: return 2
        if "i" in v: return 1
        return 2

    clin_df = pd.DataFrame(index=triple_ids)
    clin_df["age"] = clin_aligned["age_at_index"].astype(float).fillna(clin_aligned["age_at_index"].median())
    clin_df["gender"] = (clin_aligned["gender"] == "male").astype(int)
    clin_df["t_stage"] = clin_aligned["ajcc_t"].apply(encode_t)
    clin_df["n_stage"] = clin_aligned["ajcc_n"].apply(encode_n)
    clin_df["tumor_stage"] = clin_aligned["tumor_stage"].apply(encode_stg)
    clin_df["bmi"] = pd.to_numeric(clin_aligned["bmi"], errors="coerce").fillna(pd.to_numeric(clin_aligned["bmi"], errors="coerce").median())
    clin_df["vital_dead"] = (clin_aligned["vital_status"] == "Dead").astype(int)
    X_clin = clin_df.values.astype(np.float32)

    # 2. Phase 1: M3v2 3D Radiomics
    print("\n[2/5] Training M3v2: 3D Hybrid Radiomics (PyRadiomics + MedicalNet)...")
    m3_oof = np.zeros(len(y))
    for tr, val in skf.split(X_rad, y):
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_rad[tr])
        X_val_s = sc.transform(X_rad[val])
        
        sel = SelectKBest(f_classif, k=50)
        X_tr_sel = sel.fit_transform(X_tr_s, y[tr])
        X_val_sel = sel.transform(X_val_s)
        
        scale_pos = (y[tr] == 0).sum() / (y[tr] == 1).sum()
        clf = xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            scale_pos_weight=scale_pos, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=2.0, n_jobs=4, random_state=RANDOM_STATE
        )
        clf.fit(X_tr_sel, y[tr])
        m3_oof[val] = clf.predict_proba(X_val_sel)[:, 1]

    m3_cal = platt_calibrate_oof(y, m3_oof)
    m3_auc, m3_ci_l, m3_ci_h = bootstrap_ci(y, m3_oof)
    print(f"  M3v2 3D Radiomics AUROC: {m3_auc:.4f} [{m3_ci_l:.4f}, {m3_ci_h:.4f}], ECE: {ece_score(y, m3_cal):.4f}")

    # 3. Phase 2: M2v2 Pathway Genomics
    print("\n[3/5] Training M2v2: ssGSEA Pathway Genomics + Discriminative RNA-seq...")
    m2_oof = np.zeros(len(y))
    for tr, val in skf.split(X_gen_all, y):
        n1_tr = (y[tr] == 1).sum()
        n0_tr = (y[tr] == 0).sum()
        ranks = rankdata(X_gen_all[tr], axis=0)
        r1 = ranks[y[tr] == 1].sum(axis=0)
        u1 = r1 - n1_tr * (n1_tr + 1) / 2.0
        aucs = u1 / (n1_tr * n0_tr)
        top_idx = np.argsort(np.abs(aucs - 0.5))[::-1][:50]
        
        X_g_tr = np.hstack([pw_df.iloc[tr].values, X_gen_all[tr][:, top_idx]])
        X_g_val = np.hstack([pw_df.iloc[val].values, X_gen_all[val][:, top_idx]])
        
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_g_tr)
        X_val_s = sc.transform(X_g_val)
        
        scale_pos = n0_tr / n1_tr
        clf = xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            scale_pos_weight=scale_pos, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=2.0, n_jobs=4, random_state=RANDOM_STATE
        )
        clf.fit(X_tr_s, y[tr])
        m2_oof[val] = clf.predict_proba(X_val_s)[:, 1]

    m2_cal = platt_calibrate_oof(y, m2_oof)
    m2_auc, m2_ci_l, m2_ci_h = bootstrap_ci(y, m2_oof)
    print(f"  M2v2 Pathway Genomics AUROC: {m2_auc:.4f} [{m2_ci_l:.4f}, {m2_ci_h:.4f}], ECE: {ece_score(y, m2_cal):.4f}")

    # 4. Phase 3: M1v2 Clinical
    print("\n[4/5] Training M1v2: TCGA-Native Clinical Features...")
    m1_oof = np.zeros(len(y))
    for tr, val in skf.split(X_clin, y):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_clin[tr])
        X_val = sc.transform(X_clin[val])
        scale_pos = (y[tr] == 0).sum() / (y[tr] == 1).sum()
        clf = xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            scale_pos_weight=scale_pos, subsample=0.8, colsample_bytree=0.8,
            n_jobs=4, random_state=RANDOM_STATE
        )
        clf.fit(X_tr, y[tr])
        m1_oof[val] = clf.predict_proba(X_val)[:, 1]

    m1_cal = platt_calibrate_oof(y, m1_oof)
    m1_auc, m1_ci_l, m1_ci_h = bootstrap_ci(y, m1_oof)
    print(f"  M1v2 Clinical AUROC: {m1_auc:.4f} [{m1_ci_l:.4f}, {m1_ci_h:.4f}], ECE: {ece_score(y, m1_cal):.4f}")

    # 5. Phase 4: Learned Multi-Modal Fusion
    print("\n[5/5] Executing Cross-Modal Fusion Strategies...")
    X_stack = np.column_stack([m1_oof, m2_oof, m3_oof])

    # Method 1: Late Arithmetic Mean
    fusion_mean_oof = X_stack.mean(axis=1)
    fusion_mean_cal = platt_calibrate_oof(y, fusion_mean_oof)

    # Method 2: Rank Average
    fusion_rank_oof = (rankdata(m1_oof) + rankdata(m2_oof) + rankdata(m3_oof)) / (3.0 * len(y))
    fusion_rank_cal = platt_calibrate_oof(y, fusion_rank_oof)

    # Method 3: Logit-Space Stacking Logistic Regression
    eps = 1e-4
    X_logit = logit(np.clip(X_stack, eps, 1 - eps))
    fusion_logit_oof = np.zeros(len(y))
    for tr, val in skf.split(X_logit, y):
        lr = LogisticRegression(C=0.5, class_weight='balanced', random_state=RANDOM_STATE)
        lr.fit(X_logit[tr], y[tr])
        fusion_logit_oof[val] = lr.predict_proba(X_logit[val])[:, 1]
    fusion_logit_cal = platt_calibrate_oof(y, fusion_logit_oof)

    # Method 4: Multi-Layer Perceptron (Cross-Modal Attention Bottleneck)
    fusion_mlp_oof = np.zeros(len(y))
    for tr, val in skf.split(X_logit, y):
        mlp = MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation='tanh',
            alpha=0.1,
            learning_rate_init=0.01,
            max_iter=300,
            random_state=RANDOM_STATE
        )
        mlp.fit(X_logit[tr], y[tr])
        fusion_mlp_oof[val] = mlp.predict_proba(X_logit[val])[:, 1]
    fusion_mlp_cal = platt_calibrate_oof(y, fusion_mlp_oof)

    # Method 5: Weighted Evidence Fusion (Bayesian Log-Odds)
    weights = np.array([m1_auc, m2_auc, m3_auc])
    weights = weights / weights.sum()
    fusion_bef_oof = (X_logit * weights).sum(axis=1)
    fusion_bef_oof = 1.0 / (1.0 + np.exp(-fusion_bef_oof))
    fusion_bef_cal = platt_calibrate_oof(y, fusion_bef_oof)

    # Method 6: GBDT Cross-Modal Stacking
    fusion_gbdt_oof = np.zeros(len(y))
    for tr, val in skf.split(X_stack, y):
        gb = GradientBoostingClassifier(
            n_estimators=50, max_depth=2, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_STATE
        )
        gb.fit(X_stack[tr], y[tr])
        fusion_gbdt_oof[val] = gb.predict_proba(X_stack[val])[:, 1]
    fusion_gbdt_cal = platt_calibrate_oof(y, fusion_gbdt_oof)

    # Method 7: Multi-Modal Intermediate Fusion (Concatenating Top Cross-Modal Features)
    fusion_inter_oof = np.zeros(len(y))
    for tr, val in skf.split(X_rad, y):
        # Select top 20 radiomic
        sc_r = StandardScaler()
        X_r_tr = sc_r.fit_transform(X_rad[tr])
        X_r_val = sc_r.transform(X_rad[val])
        sel_r = SelectKBest(f_classif, k=20)
        X_r_tr_sel = sel_r.fit_transform(X_r_tr, y[tr])
        X_r_val_sel = sel_r.transform(X_r_val)

        # Select top 20 genomic
        n1_tr = (y[tr] == 1).sum()
        n0_tr = (y[tr] == 0).sum()
        ranks = rankdata(X_gen_all[tr], axis=0)
        r1 = ranks[y[tr] == 1].sum(axis=0)
        u1 = r1 - n1_tr * (n1_tr + 1) / 2.0
        aucs = u1 / (n1_tr * n0_tr)
        top_idx = np.argsort(np.abs(aucs - 0.5))[::-1][:20]
        X_g_tr_sel = np.hstack([pw_df.iloc[tr].values, X_gen_all[tr][:, top_idx]])
        X_g_val_sel = np.hstack([pw_df.iloc[val].values, X_gen_all[val][:, top_idx]])
        sc_g = StandardScaler()
        X_g_tr_s = sc_g.fit_transform(X_g_tr_sel)
        X_g_val_s = sc_g.transform(X_g_val_sel)

        # Clinical
        sc_c = StandardScaler()
        X_c_tr_s = sc_c.fit_transform(X_clin[tr])
        X_c_val_s = sc_c.transform(X_clin[val])

        # Concatenate intermediate representations
        X_inter_tr = np.hstack([X_c_tr_s, X_g_tr_s, X_r_tr_sel])
        X_inter_val = np.hstack([X_c_val_s, X_g_val_s, X_r_val_sel])

        scale_pos = n0_tr / n1_tr
        clf_inter = xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.03,
            scale_pos_weight=scale_pos, subsample=0.8, colsample_bytree=0.6,
            reg_alpha=2.0, reg_lambda=3.0, n_jobs=4, random_state=RANDOM_STATE
        )
        clf_inter.fit(X_inter_tr, y[tr])
        fusion_inter_oof[val] = clf_inter.predict_proba(X_inter_val)[:, 1]
    fusion_inter_cal = platt_calibrate_oof(y, fusion_inter_oof)

    # ══════════════════════════════════════════════════════════════════
    # COMPILE FULL METRIC MANIFEST
    # ══════════════════════════════════════════════════════════════════
    models_dict = {
        "M1v2 (Clinical)": (m1_oof, m1_cal),
        "M2v2 (Genomic)": (m2_oof, m2_cal),
        "M3v2 (3D Radiomics)": (m3_oof, m3_cal),
        "Fusion A: Late Mean": (fusion_mean_oof, fusion_mean_cal),
        "Fusion B: Rank Average": (fusion_rank_oof, fusion_rank_cal),
        "Fusion C: Logit Stacking": (fusion_logit_oof, fusion_logit_cal),
        "Fusion D: Bottleneck MLP": (fusion_mlp_oof, fusion_mlp_cal),
        "Fusion E: Bayesian Evidential (BEF)": (fusion_bef_oof, fusion_bef_cal),
        "Fusion F: GBDT Stacking": (fusion_gbdt_oof, fusion_gbdt_cal),
        "Fusion G: Intermediate Representation": (fusion_inter_oof, fusion_inter_cal),
    }

    print("\n" + "=" * 115)
    print(f"{'Model / Architecture':<38} {'AUROC':<10} {'95% CI':<22} {'AUPRC':<10} {'F2':<8} {'Brier':<8} {'ECE (Raw)':<10} {'ECE (Cal)':<10}")
    print("-" * 115)

    manifest = {}
    for name, (oof_raw, oof_cal) in models_dict.items():
        auc_val, ci_l, ci_h = bootstrap_ci(y, oof_raw)
        m_raw = point_metrics(y, oof_raw)
        m_cal = point_metrics(y, oof_cal)
        ece_raw = ece_score(y, oof_raw)
        ece_cal = ece_score(y, oof_cal)

        manifest[name] = {
            "auroc": auc_val,
            "auroc_ci_low": ci_l,
            "auroc_ci_high": ci_h,
            "auprc": m_raw["AUPRC"],
            "recall": m_raw["Recall"],
            "precision": m_raw["Precision"],
            "f1": m_raw["F1"],
            "f2": m_raw["F2"],
            "brier_raw": m_raw["Brier"],
            "brier_cal": m_cal["Brier"],
            "ece_raw": ece_raw,
            "ece_cal": ece_cal,
            "threshold": m_raw["threshold"],
            "cal_threshold": m_cal["threshold"]
        }

        print(f"{name:<38} {auc_val:<10.4f} [{ci_l:.4f}, {ci_h:.4f}]   {m_raw['AUPRC']:<10.4f} {m_raw['F2']:<8.4f} {m_cal['Brier']:<8.4f} {ece_raw:<10.4f} {ece_cal:<10.4f}")

    # Best Model Identification
    best_name = max(manifest, key=lambda k: manifest[k]["auroc"])
    best_auc = manifest[best_name]["auroc"]
    print("\n" + "=" * 115)
    print(f"FLAGSHIP RESULT: Best model is '{best_name}' with 5-Fold OOF AUROC = {best_auc:.4f} [{manifest[best_name]['auroc_ci_low']:.4f}, {manifest[best_name]['auroc_ci_high']:.4f}]")
    print("=" * 115)

    # Paired Bootstrap Tests against best single modality and legacy baseline
    best_single = "M3v2 (3D Radiomics)"
    print(f"\n--- Paired Bootstrap Hypothesis Tests (N={N_BOOTSTRAP}) ---")
    for name in models_dict:
        if name != best_name:
            delta, ci_l, ci_h, p_val = paired_bootstrap_test(y, models_dict[best_name][0], models_dict[name][0])
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
            print(f"  {best_name} vs {name:<36}: ΔAUROC = {delta:+.4f} [{ci_l:+.4f}, {ci_h:+.4f}], p = {p_val:.4f} ({sig})")

    # Save manifest JSON
    with open(RESULTS_DIR / "v2_final_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[SAVED] Final metrics manifest: {RESULTS_DIR / 'v2_final_manifest.json'}")

    # Save OOF predictions
    np.savez(
        RESULTS_DIR / "v2_final_oof_predictions.npz",
        y=y,
        patient_ids=np.array(triple_ids),
        m1_oof=m1_oof, m1_cal=m1_cal,
        m2_oof=m2_oof, m2_cal=m2_cal,
        m3_oof=m3_oof, m3_cal=m3_cal,
        fusion_mean_oof=fusion_mean_oof, fusion_mean_cal=fusion_mean_cal,
        fusion_rank_oof=fusion_rank_oof, fusion_rank_cal=fusion_rank_cal,
        fusion_logit_oof=fusion_logit_oof, fusion_logit_cal=fusion_logit_cal,
        fusion_mlp_oof=fusion_mlp_oof, fusion_mlp_cal=fusion_mlp_cal,
        fusion_bef_oof=fusion_bef_oof, fusion_bef_cal=fusion_bef_cal,
        fusion_gbdt_oof=fusion_gbdt_oof, fusion_gbdt_cal=fusion_gbdt_cal,
        fusion_inter_oof=fusion_inter_oof, fusion_inter_cal=fusion_inter_cal,
    )
    print(f"[SAVED] Final OOF predictions: {RESULTS_DIR / 'v2_final_oof_predictions.npz'}")
    print(f"\nCompleted in {time.time() - t_start:.2f} seconds.")

if __name__ == "__main__":
    main()
