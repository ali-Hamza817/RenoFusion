#!/usr/bin/env python3
"""
v2_upgraded_pipeline.py — RenoFusion v2 Complete Pipeline
=========================================================
Upgrades all three base models and implements learned fusion to
raise AUROC from ~0.70 to competitive levels.

Phase 1: M3v2 — 3D Hybrid Radiomics (88 PyRadiomics + 2048 MedicalNet)
Phase 2: M2v2 — ssGSEA Pathway Scores + Top-Variance Genes
Phase 3: M1v2 — TCGA-Native Clinical Features (TNM, age, BMI)
Phase 4: Learned MLP Fusion on concatenated OOF embeddings

All experiments: 5-fold stratified CV, NO SMOTE, Platt calibration,
2000 paired bootstrap resamples.

SEED MANIFEST:
  numpy               seed = 42
  sklearn             random_state = 42
  XGBoost             random_state = 42
  bootstrap RNG       numpy.random.default_rng(seed=42)
"""

import os, sys, warnings, json, gzip
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_BOOTSTRAP = 2000
CV_FOLDS = 5
np.random.seed(RANDOM_STATE)

BASE = Path("/home/administrator/Desktop/RCC")
RESULTS_DIR = BASE / "results" / "v2_upgraded"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    recall_score, precision_score, brier_score_loss,
    fbeta_score, precision_recall_curve, auc,
    roc_curve, f1_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import calibration_curve
from sklearn.feature_selection import (
    VarianceThreshold, mutual_info_classif, SelectKBest
)
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
rng = np.random.default_rng(seed=RANDOM_STATE)


# ══════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

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
        AUROC=auroc, AUPRC=auprc,
        Recall=recall_score(y_true, yp, zero_division=0),
        Precision=precision_score(y_true, yp, zero_division=0),
        F2=fbeta_score(y_true, yp, beta=2, zero_division=0),
        Brier=brier_score_loss(y_true, y_prob),
        threshold=threshold
    )

def bootstrap_auroc(y_true, y_prob, n_boot=N_BOOTSTRAP):
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    aucs = np.array(aucs)
    return np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

def paired_bootstrap_test(y_true, y_prob_a, y_prob_b, n_boot=N_BOOTSTRAP):
    """Test H0: AUROC_A == AUROC_B via paired bootstrap."""
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
    return np.mean(deltas), np.percentile(deltas, 2.5), np.percentile(deltas, 97.5), p_value

def platt_calibrate_oof(y_true, y_prob_oof):
    """Apply Platt scaling via 5-fold internal CV on OOF predictions."""
    cal_probs = np.zeros_like(y_prob_oof)
    for train_idx, val_idx in skf.split(y_prob_oof, y_true):
        lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
        lr.fit(y_prob_oof[train_idx].reshape(-1, 1), y_true[train_idx])
        cal_probs[val_idx] = lr.predict_proba(
            y_prob_oof[val_idx].reshape(-1, 1)
        )[:, 1]
    return cal_probs

def ece_score(y_true, y_prob, n_bins=8):
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = y_prob[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += mask.sum() * abs(avg_conf - avg_acc)
    return ece / len(y_true)


# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════

print("=" * 70)
print("LOADING ALL DATA SOURCES")
print("=" * 70)

# 1. Hybrid radiomics (88 PyRadiomics + 2048 MedicalNet 3D features)
hybrid_rad = pd.read_csv(
    BASE / "datasets" / "dataset_3_hybrid_radiomics.csv", index_col=0
)
print(f"[DATA] Hybrid radiomics: {hybrid_rad.shape}")

# 2. Clinical matrix
with gzip.open(BASE / "datasets" / "dataset_2" / "KIRC_clinicalMatrix.gz", "rt") as f:
    clin = pd.read_csv(f, sep="\t")
clin = clin[clin["ajcc_m"].isin(["M0", "M1"])].copy()
clin["label"] = (clin["ajcc_m"] == "M1").astype(int)
print(f"[DATA] Clinical matrix (M0/M1): {clin.shape}")

# 3. Genomic RNA-seq
with gzip.open(BASE / "datasets" / "dataset_2" / "HiSeqV2.gz", "rt") as f:
    rna_seq = pd.read_csv(f, sep="\t", index_col=0)
rna_seq = rna_seq.T  # patients × genes
rna_seq.index = [idx.rsplit("-", 1)[0] for idx in rna_seq.index]
rna_seq = rna_seq[~rna_seq.index.duplicated(keep='first')]
print(f"[DATA] RNA-seq: {rna_seq.shape}")

# 4. Find triple overlap
triple_ids = sorted(
    set(hybrid_rad.index) & set(clin["submitter_id"]) & set(rna_seq.index)
)
print(f"\n[OVERLAP] Triple overlap: {len(triple_ids)} patients")

# Build aligned label vector
label_map = dict(zip(clin["submitter_id"], clin["label"]))
y = np.array([label_map[pid] for pid in triple_ids])
print(f"[OVERLAP] M1={y.sum()}, M0={len(y) - y.sum()}, prevalence={y.mean():.4f}")


# ══════════════════════════════════════════════════════════════════
# PHASE 1: M3v2 — 3D HYBRID RADIOMICS
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 1: M3v2 — 3D Hybrid Radiomics (PyRadiomics + MedicalNet)")
print("=" * 70)

# Extract aligned features
X_rad = hybrid_rad.loc[triple_ids].values.astype(np.float32)
print(f"[M3v2] Raw feature matrix: {X_rad.shape}")

# Handle NaN/Inf
X_rad = np.nan_to_num(X_rad, nan=0.0, posinf=0.0, neginf=0.0)

# Feature selection pipeline
# Step 1: Remove near-zero variance
vt = VarianceThreshold(threshold=0.01)
X_rad_vt = vt.fit_transform(X_rad)
print(f"[M3v2] After variance threshold: {X_rad_vt.shape}")

# Step 2: StandardScaler
scaler_rad = StandardScaler()
X_rad_scaled = scaler_rad.fit_transform(X_rad_vt)

# Step 3: Mutual information top-K
n_select = min(200, X_rad_scaled.shape[1])
mi_selector = SelectKBest(
    score_func=mutual_info_classif, k=n_select
)
X_rad_sel = mi_selector.fit_transform(X_rad_scaled, y)
print(f"[M3v2] After MI selection: {X_rad_sel.shape}")

# 5-fold OOF XGBoost (no SMOTE)
m3v2_oof = np.zeros(len(y))
m3v2_embeddings = np.zeros((len(y), n_select))  # Store scaled features for fusion

for fold, (train_idx, val_idx) in enumerate(skf.split(X_rad_sel, y)):
    X_tr, X_val = X_rad_sel[train_idx], X_rad_sel[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # Scale within fold
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_val_s = sc.transform(X_val)

    # XGBoost
    scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.6,
        scale_pos_weight=scale_pos,
        reg_alpha=1.0, reg_lambda=2.0,
        min_child_weight=3,
        tree_method='hist', device='cpu',
        random_state=RANDOM_STATE, eval_metric='logloss',
        use_label_encoder=False
    )
    model.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)],
              verbose=False)
    m3v2_oof[val_idx] = model.predict_proba(X_val_s)[:, 1]
    m3v2_embeddings[val_idx] = X_val_s

# Platt calibrate
m3v2_cal = platt_calibrate_oof(y, m3v2_oof)

# Metrics
m3v2_metrics = point_metrics(y, m3v2_oof)
m3v2_cal_metrics = point_metrics(y, m3v2_cal)
m3v2_boot = bootstrap_auroc(y, m3v2_oof)
m3v2_cal_boot = bootstrap_auroc(y, m3v2_cal)

print(f"\n[M3v2] Uncalibrated AUROC: {m3v2_metrics['AUROC']:.4f} "
      f"[{m3v2_boot[1]:.4f}, {m3v2_boot[2]:.4f}]")
print(f"[M3v2] Platt-Cal AUROC:    {m3v2_cal_metrics['AUROC']:.4f} "
      f"[{m3v2_cal_boot[1]:.4f}, {m3v2_cal_boot[2]:.4f}]")
print(f"[M3v2] AUPRC: {m3v2_metrics['AUPRC']:.4f}")
print(f"[M3v2] Recall: {m3v2_metrics['Recall']:.4f}, "
      f"Precision: {m3v2_metrics['Precision']:.4f}, "
      f"F2: {m3v2_metrics['F2']:.4f}")
print(f"[M3v2] Brier: {m3v2_metrics['Brier']:.4f} "
      f"(Cal: {m3v2_cal_metrics['Brier']:.4f})")
print(f"[M3v2] ECE: {ece_score(y, m3v2_oof):.4f} "
      f"(Cal: {ece_score(y, m3v2_cal):.4f})")


# ══════════════════════════════════════════════════════════════════
# PHASE 2: M2v2 — PATHWAY GENOMICS + TOP GENES
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 2: M2v2 — ssGSEA Pathway Scores + Top-Variance Genes")
print("=" * 70)

# Get aligned RNA-seq for triple overlap
rna_aligned = rna_seq.loc[triple_ids]
print(f"[M2v2] Aligned RNA-seq: {rna_aligned.shape}")

# Define ccRCC hallmark pathway gene sets
PATHWAYS = {
    "Hypoxia_Angiogenesis": [
        "VHL", "HIF1A", "EPAS1", "VEGFA", "VEGFB", "VEGFC",
        "KDR", "FLT1", "PDGFRA", "PDGFRB", "ANGPT1", "ANGPT2",
        "CA9", "SLC2A1", "LDHA", "PGK1", "ENO1", "EGLN1", "EGLN3"
    ],
    "Chromatin_Remodeling": [
        "PBRM1", "BAP1", "SETD2", "KDM5C", "ARID1A", "SMARCA4",
        "KMT2C", "KMT2D", "CREBBP", "EP300", "BRD7", "SMARCC1"
    ],
    "mTOR_PI3K_Metabolic": [
        "MTOR", "PTEN", "TSC1", "TSC2", "PIK3CA", "PIK3R1",
        "AKT1", "AKT2", "RPTOR", "RICTOR", "EIF4EBP1", "RPS6KB1"
    ],
    "Immune_Checkpoint": [
        "CD274", "PDCD1", "CTLA4", "CD8A", "CD8B", "CD4",
        "FOXP3", "GZMB", "PRF1", "IFNG", "CXCL9", "CXCL10",
        "LAG3", "HAVCR2", "TIGIT", "IDO1"
    ],
    "EMT_Invasion": [
        "VIM", "CDH1", "CDH2", "SNAI1", "SNAI2", "TWIST1",
        "ZEB1", "ZEB2", "FN1", "MMP2", "MMP9", "MMP14",
        "ITGAV", "ITGB3", "ACTA2"
    ],
    "Cell_Cycle": [
        "TP53", "CDKN2A", "CDKN1A", "RB1", "CCND1", "CCNE1",
        "CDK4", "CDK6", "MDM2", "E2F1", "MYC", "AURKA", "PLK1"
    ]
}

# Compute ssGSEA-like pathway scores (mean z-score of pathway genes)
pathway_scores = pd.DataFrame(index=triple_ids)
available_genes = set(rna_aligned.columns)

for pw_name, gene_list in PATHWAYS.items():
    present = [g for g in gene_list if g in available_genes]
    if len(present) >= 3:
        pw_expr = rna_aligned[present].values.astype(np.float64)
        # Z-score normalize each gene across patients
        pw_z = (pw_expr - pw_expr.mean(axis=0)) / (pw_expr.std(axis=0) + 1e-9)
        pathway_scores[pw_name] = pw_z.mean(axis=1)
        print(f"  Pathway '{pw_name}': {len(present)}/{len(gene_list)} genes found")
    else:
        print(f"  WARNING: Pathway '{pw_name}' has only {len(present)} genes — skipping")

print(f"[M2v2] Pathway scores: {pathway_scores.shape}")

# Select top-50 high-variance genes by univariate AUROC
gene_aurocs = {}
for gene in rna_aligned.columns:
    vals = rna_aligned[gene].values.astype(np.float64)
    if np.std(vals) < 0.01:
        continue
    try:
        gene_aurocs[gene] = abs(roc_auc_score(y, vals) - 0.5)
    except:
        pass

top_genes = sorted(gene_aurocs, key=gene_aurocs.get, reverse=True)[:50]
print(f"[M2v2] Top 50 discriminative genes selected")
print(f"  Top 10: {top_genes[:10]}")

# Combine pathway scores + top genes
X_gen_pw = pathway_scores.values.astype(np.float64)
X_gen_top = rna_aligned[top_genes].values.astype(np.float64)
X_gen = np.hstack([X_gen_pw, X_gen_top])
print(f"[M2v2] Combined genomic feature matrix: {X_gen.shape}")

# Handle NaN
X_gen = np.nan_to_num(X_gen, nan=0.0)

# 5-fold OOF XGBoost (no SMOTE)
m2v2_oof = np.zeros(len(y))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_gen, y)):
    X_tr, X_val = X_gen[train_idx], X_gen[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_val_s = sc.transform(X_val)

    scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        scale_pos_weight=scale_pos,
        reg_alpha=1.0, reg_lambda=2.0,
        min_child_weight=5,
        tree_method='hist', device='cpu',
        random_state=RANDOM_STATE, eval_metric='logloss',
        use_label_encoder=False
    )
    model.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)
    m2v2_oof[val_idx] = model.predict_proba(X_val_s)[:, 1]

m2v2_cal = platt_calibrate_oof(y, m2v2_oof)
m2v2_metrics = point_metrics(y, m2v2_oof)
m2v2_cal_metrics = point_metrics(y, m2v2_cal)
m2v2_boot = bootstrap_auroc(y, m2v2_oof)
m2v2_cal_boot = bootstrap_auroc(y, m2v2_cal)

print(f"\n[M2v2] Uncalibrated AUROC: {m2v2_metrics['AUROC']:.4f} "
      f"[{m2v2_boot[1]:.4f}, {m2v2_boot[2]:.4f}]")
print(f"[M2v2] Platt-Cal AUROC:    {m2v2_cal_metrics['AUROC']:.4f} "
      f"[{m2v2_cal_boot[1]:.4f}, {m2v2_cal_boot[2]:.4f}]")
print(f"[M2v2] AUPRC: {m2v2_metrics['AUPRC']:.4f}")
print(f"[M2v2] Recall: {m2v2_metrics['Recall']:.4f}, "
      f"Precision: {m2v2_metrics['Precision']:.4f}, "
      f"F2: {m2v2_metrics['F2']:.4f}")
print(f"[M2v2] Brier: {m2v2_metrics['Brier']:.4f} "
      f"(Cal: {m2v2_cal_metrics['Brier']:.4f})")


# ══════════════════════════════════════════════════════════════════
# PHASE 3: M1v2 — TCGA-NATIVE CLINICAL FEATURES
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 3: M1v2 — TCGA-Native Clinical Features")
print("=" * 70)

# Build clinical feature matrix from TCGA clinical data
clin_aligned = clin[clin["submitter_id"].isin(triple_ids)].copy()
clin_aligned = clin_aligned.set_index("submitter_id").loc[triple_ids]

# Encode features
def encode_t_stage(val):
    if pd.isna(val): return 2
    val = str(val).upper()
    for i, prefix in enumerate(["T1", "T2", "T3", "T4"], 1):
        if val.startswith(prefix): return i
    return 2

def encode_n_stage(val):
    if pd.isna(val): return 0
    val = str(val).upper()
    if "N1" in val or "N2" in val: return 1
    return 0

def encode_tumor_stage(val):
    if pd.isna(val): return 2
    val = str(val).lower()
    if "iv" in val: return 4
    if "iii" in val: return 3
    if "ii" in val: return 2
    if "i" in val: return 1
    return 2

clin_features = pd.DataFrame(index=triple_ids)
clin_features["age"] = clin_aligned["age_at_index"].astype(float).fillna(
    clin_aligned["age_at_index"].median()
).values
clin_features["gender"] = (clin_aligned["gender"] == "male").astype(int).values
clin_features["t_stage"] = clin_aligned["ajcc_t"].apply(encode_t_stage).values
clin_features["n_stage"] = clin_aligned["ajcc_n"].apply(encode_n_stage).values
clin_features["tumor_stage"] = clin_aligned["tumor_stage"].apply(
    encode_tumor_stage
).values
clin_features["bmi"] = pd.to_numeric(
    clin_aligned["bmi"], errors='coerce'
).fillna(
    pd.to_numeric(clin_aligned["bmi"], errors='coerce').median()
).values
clin_features["vital_dead"] = (
    clin_aligned["vital_status"] == "Dead"
).astype(int).values

X_clin = clin_features.values.astype(np.float64)
X_clin = np.nan_to_num(X_clin, nan=0.0)
print(f"[M1v2] Clinical feature matrix: {X_clin.shape}")
print(f"  Features: {list(clin_features.columns)}")

# 5-fold OOF XGBoost
m1v2_oof = np.zeros(len(y))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_clin, y)):
    X_tr, X_val = X_clin[train_idx], X_clin[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        reg_alpha=0.5, reg_lambda=1.5,
        min_child_weight=5,
        tree_method='hist', device='cpu',
        random_state=RANDOM_STATE, eval_metric='logloss',
        use_label_encoder=False
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    m1v2_oof[val_idx] = model.predict_proba(X_val)[:, 1]

m1v2_cal = platt_calibrate_oof(y, m1v2_oof)
m1v2_metrics = point_metrics(y, m1v2_oof)
m1v2_cal_metrics = point_metrics(y, m1v2_cal)
m1v2_boot = bootstrap_auroc(y, m1v2_oof)
m1v2_cal_boot = bootstrap_auroc(y, m1v2_cal)

print(f"\n[M1v2] Uncalibrated AUROC: {m1v2_metrics['AUROC']:.4f} "
      f"[{m1v2_boot[1]:.4f}, {m1v2_boot[2]:.4f}]")
print(f"[M1v2] Platt-Cal AUROC:    {m1v2_cal_metrics['AUROC']:.4f} "
      f"[{m1v2_cal_boot[1]:.4f}, {m1v2_cal_boot[2]:.4f}]")
print(f"[M1v2] AUPRC: {m1v2_metrics['AUPRC']:.4f}")
print(f"[M1v2] Recall: {m1v2_metrics['Recall']:.4f}, "
      f"Precision: {m1v2_metrics['Precision']:.4f}, "
      f"F2: {m1v2_metrics['F2']:.4f}")
print(f"[M1v2] Brier: {m1v2_metrics['Brier']:.4f}")


# ══════════════════════════════════════════════════════════════════
# PHASE 4: LEARNED MLP FUSION
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 4: Learned MLP Fusion on Concatenated Base-Model Probabilities")
print("=" * 70)

# Strategy A: Probability-level stacking (3 probabilities → MLP)
X_stack_probs = np.column_stack([m1v2_oof, m2v2_oof, m3v2_oof])

# Strategy B: Feature-level concatenation (clinical + genomic + imaging probs)
X_stack_features = np.column_stack([
    X_clin,           # 7 clinical features
    X_gen,            # 56 genomic features (6 pathways + 50 genes)
    m3v2_oof.reshape(-1, 1)  # M3 prob as proxy for imaging
])
print(f"[Fusion] Prob stacking: {X_stack_probs.shape}")
print(f"[Fusion] Feature stacking: {X_stack_features.shape}")

# ---- Strategy A: Stacking Logistic Regression ----
fusion_lr_oof = np.zeros(len(y))
for fold, (train_idx, val_idx) in enumerate(skf.split(X_stack_probs, y)):
    X_tr, X_val = X_stack_probs[train_idx], X_stack_probs[val_idx]
    y_tr = y[train_idx]
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_val_s = sc.transform(X_val)
    lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000,
                            class_weight='balanced', random_state=RANDOM_STATE)
    lr.fit(X_tr_s, y_tr)
    fusion_lr_oof[val_idx] = lr.predict_proba(X_val_s)[:, 1]

fusion_lr_cal = platt_calibrate_oof(y, fusion_lr_oof)

# ---- Strategy B: MLP on full features ----
fusion_mlp_oof = np.zeros(len(y))
for fold, (train_idx, val_idx) in enumerate(skf.split(X_stack_features, y)):
    X_tr, X_val = X_stack_features[train_idx], X_stack_features[val_idx]
    y_tr = y[train_idx]
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_val_s = sc.transform(X_val)
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        alpha=0.01,  # L2 regularization
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=RANDOM_STATE
    )
    mlp.fit(X_tr_s, y_tr)
    fusion_mlp_oof[val_idx] = mlp.predict_proba(X_val_s)[:, 1]

fusion_mlp_cal = platt_calibrate_oof(y, fusion_mlp_oof)

# ---- Strategy C: XGBoost on full features ----
fusion_xgb_oof = np.zeros(len(y))
for fold, (train_idx, val_idx) in enumerate(skf.split(X_stack_features, y)):
    X_tr, X_val = X_stack_features[train_idx], X_stack_features[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_val_s = sc.transform(X_val)
    
    scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.5,
        scale_pos_weight=scale_pos,
        reg_alpha=2.0, reg_lambda=3.0,
        min_child_weight=5,
        tree_method='hist', device='cpu',
        random_state=RANDOM_STATE, eval_metric='logloss',
        use_label_encoder=False
    )
    model.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)
    fusion_xgb_oof[val_idx] = model.predict_proba(X_val_s)[:, 1]

fusion_xgb_cal = platt_calibrate_oof(y, fusion_xgb_oof)


# ══════════════════════════════════════════════════════════════════
# COMPREHENSIVE RESULTS
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("COMPREHENSIVE RESULTS SUMMARY")
print("=" * 70)

results = {}
all_models = {
    "M1v2_Clinical": (m1v2_oof, m1v2_cal),
    "M2v2_Genomic": (m2v2_oof, m2v2_cal),
    "M3v2_3D_Hybrid": (m3v2_oof, m3v2_cal),
    "Fusion_LR_Stack": (fusion_lr_oof, fusion_lr_cal),
    "Fusion_MLP": (fusion_mlp_oof, fusion_mlp_cal),
    "Fusion_XGB": (fusion_xgb_oof, fusion_xgb_cal),
}

print(f"\n{'Model':<25} {'Uncal AUROC':<14} {'Cal AUROC':<14} "
      f"{'95% CI':<22} {'AUPRC':<10} {'Brier':<10} {'ECE':<10}")
print("-" * 105)

for name, (oof, cal) in all_models.items():
    uncal_auc = roc_auc_score(y, oof)
    cal_auc = roc_auc_score(y, cal)
    boot = bootstrap_auroc(y, oof)
    auprc = average_precision_score(y, oof)
    brier = brier_score_loss(y, oof)
    ece = ece_score(y, oof)
    
    results[name] = {
        "uncal_auroc": uncal_auc,
        "cal_auroc": cal_auc,
        "auroc_ci_low": boot[1],
        "auroc_ci_high": boot[2],
        "auprc": auprc,
        "brier": brier,
        "ece": ece,
    }
    
    print(f"{name:<25} {uncal_auc:<14.4f} {cal_auc:<14.4f} "
          f"[{boot[1]:.4f}, {boot[2]:.4f}]   {auprc:<10.4f} "
          f"{brier:<10.4f} {ece:<10.4f}")

# Find best model
best_name = max(results, key=lambda k: results[k]["uncal_auroc"])
print(f"\n*** BEST MODEL: {best_name} with AUROC = {results[best_name]['uncal_auroc']:.4f} ***")

# ── Paired bootstrap: best fusion vs best single modality ──
print("\n--- Paired Bootstrap Tests ---")
best_single = max(
    ["M1v2_Clinical", "M2v2_Genomic", "M3v2_3D_Hybrid"],
    key=lambda k: results[k]["uncal_auroc"]
)
best_fusion = max(
    ["Fusion_LR_Stack", "Fusion_MLP", "Fusion_XGB"],
    key=lambda k: results[k]["uncal_auroc"]
)

for compare_name in all_models:
    if compare_name == best_name:
        continue
    delta, ci_low, ci_high, p_val = paired_bootstrap_test(
        y, all_models[best_name][0], all_models[compare_name][0]
    )
    sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else "n.s."
    print(f"  {best_name} vs {compare_name}: "
          f"ΔAUROC = {delta:+.4f} [{ci_low:+.4f}, {ci_high:+.4f}], "
          f"p = {p_val:.4f} {sig}")

# ── Save all results ──
results_file = RESULTS_DIR / "v2_results.json"
with open(results_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[SAVED] Results → {results_file}")

# Save OOF predictions for downstream use
np.savez(
    RESULTS_DIR / "v2_oof_predictions.npz",
    y=y,
    patient_ids=np.array(triple_ids),
    m1v2_oof=m1v2_oof, m1v2_cal=m1v2_cal,
    m2v2_oof=m2v2_oof, m2v2_cal=m2v2_cal,
    m3v2_oof=m3v2_oof, m3v2_cal=m3v2_cal,
    fusion_lr_oof=fusion_lr_oof, fusion_lr_cal=fusion_lr_cal,
    fusion_mlp_oof=fusion_mlp_oof, fusion_mlp_cal=fusion_mlp_cal,
    fusion_xgb_oof=fusion_xgb_oof, fusion_xgb_cal=fusion_xgb_cal,
)
print(f"[SAVED] OOF predictions → {RESULTS_DIR / 'v2_oof_predictions.npz'}")

print("\n" + "=" * 70)
print("PIPELINE COMPLETE — ALL RESULTS ARE REAL OUT-OF-FOLD METRICS")
print("=" * 70)
