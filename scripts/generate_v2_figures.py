#!/usr/bin/env python3
"""
generate_v2_figures.py — Publication Figure Generator for RenoFusion v2
========================================================================
Generates publication-quality figures:
1. Fig 2: ROC Curves (Single Modalities vs Fusion Architectures)
2. Fig 3: Precision-Recall Curves (PR-AUC Comparison)
3. Fig 4: Calibration Curves & Reliability Diagrams (Raw vs Platt-Calibrated)
4. Fig 5: Multi-Modal Performance Barplot (AUROC, AUPRC, F2, ECE)
"""

import os, json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.calibration import calibration_curve

BASE = Path("/home/administrator/Desktop/RCC")
RESULTS_DIR = BASE / "results" / "v2_upgraded"
FIGS_DIR = BASE / "paper" / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
with open(RESULTS_DIR / "v2_final_manifest.json") as f:
    manifest = json.load(f)

oof_data = np.load(RESULTS_DIR / "v2_final_oof_predictions.npz")
y = oof_data["y"]

# Setup styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

colors = {
    'M1v2 (Clinical)': '#2b5c8f',
    'M2v2 (Genomic)': '#d95f02',
    'M3v2 (3D Radiomics)': '#7570b3',
    'Fusion A: Late Mean': '#e7298a',
    'Fusion B: Rank Average': '#66a61e',
    'Fusion C: Logit Stacking': '#e6ab02',
    'Fusion D: Bottleneck MLP': '#a6761d',
    'Fusion E: Bayesian Evidential (BEF)': '#d62728', # Flagship Highlight
}

# -------------------------------------------------------------
# FIGURE 2: ROC Curves
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 6.5))

# Plot single modalities (dashed)
for name, key in [
    ('M1v2 (Clinical)', 'm1_oof'),
    ('M2v2 (Genomic)', 'm2_oof'),
    ('M3v2 (3D Radiomics)', 'm3_oof')
]:
    prob = oof_data[key]
    fpr, tpr, _ = roc_curve(y, prob)
    auroc = manifest[name]['auroc']
    ci_l = manifest[name]['auroc_ci_low']
    ci_h = manifest[name]['auroc_ci_high']
    ax.plot(fpr, tpr, label=f"{name} (AUROC = {auroc:.3f} [{ci_l:.2f}-{ci_h:.2f}])",
            color=colors[name], linestyle='--', linewidth=1.8, alpha=0.85)

# Plot fusions (solid)
for name, key, lw, alpha in [
    ('Fusion A: Late Mean', 'fusion_mean_oof', 1.8, 0.8),
    ('Fusion B: Rank Average', 'fusion_rank_oof', 1.8, 0.8),
    ('Fusion C: Logit Stacking', 'fusion_logit_oof', 1.8, 0.8),
    ('Fusion D: Bottleneck MLP', 'fusion_mlp_oof', 1.8, 0.8),
    ('Fusion E: Bayesian Evidential (BEF)', 'fusion_bef_oof', 2.8, 1.0)
]:
    prob = oof_data[key]
    fpr, tpr, _ = roc_curve(y, prob)
    auroc = manifest[name]['auroc']
    ci_l = manifest[name]['auroc_ci_low']
    ci_h = manifest[name]['auroc_ci_high']
    ax.plot(fpr, tpr, label=f"{name} (AUROC = {auroc:.3f} [{ci_l:.2f}-{ci_h:.2f}])",
            color=colors[name], linestyle='-', linewidth=lw, alpha=alpha)

ax.plot([0, 1], [0, 1], 'k:', alpha=0.5, label='Chance (AUROC = 0.500)')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.set_xlabel('False Positive Rate (1 - Specificity)')
ax.set_ylabel('True Positive Rate (Sensitivity)')
ax.set_title('Out-of-Fold ROC Curves (5-Fold Stratified CV, N=126 ccRCC Cohort)')
ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.92)
ax.grid(True, linestyle='--', alpha=0.4)
fig.savefig(FIGS_DIR / "fig2_roc_curves.pdf")
fig.savefig(FIGS_DIR / "fig2_roc_curves.png")
plt.close(fig)
print("[FIG] Saved fig2_roc_curves")

# -------------------------------------------------------------
# FIGURE 3: Precision-Recall Curves
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 6.5))
for name, key in [
    ('M1v2 (Clinical)', 'm1_oof'),
    ('M2v2 (Genomic)', 'm2_oof'),
    ('M3v2 (3D Radiomics)', 'm3_oof'),
    ('Fusion A: Late Mean', 'fusion_mean_oof'),
    ('Fusion B: Rank Average', 'fusion_rank_oof'),
    ('Fusion C: Logit Stacking', 'fusion_logit_oof'),
    ('Fusion D: Bottleneck MLP', 'fusion_mlp_oof'),
    ('Fusion E: Bayesian Evidential (BEF)', 'fusion_bef_oof')
]:
    prob = oof_data[key]
    prec, rec, _ = precision_recall_curve(y, prob)
    auprc = manifest[name]['auprc']
    lw = 2.8 if 'BEF' in name else 1.8
    ls = '--' if 'M1' in name or 'M2' in name or 'M3' in name else '-'
    ax.plot(rec, prec, label=f"{name} (AUPRC = {auprc:.3f})",
            color=colors[name], linestyle=ls, linewidth=lw)

baseline_prevalence = y.mean()
ax.axhline(baseline_prevalence, color='k', linestyle=':', label=f'Prevalence Baseline ({baseline_prevalence:.3f})')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.set_xlabel('Recall (Sensitivity)')
ax.set_ylabel('Precision (Positive Predictive Value)')
ax.set_title('Out-of-Fold Precision-Recall Curves (5-Fold Stratified CV)')
ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.92)
ax.grid(True, linestyle='--', alpha=0.4)
fig.savefig(FIGS_DIR / "fig3_pr_curves.pdf")
fig.savefig(FIGS_DIR / "fig3_pr_curves.png")
plt.close(fig)
print("[FIG] Saved fig3_pr_curves")

# -------------------------------------------------------------
# FIGURE 4: Calibration Reliability Diagrams
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

eval_models = [
    ('M3v2 (3D Radiomics)', 'm3_oof', 'm3_cal'),
    ('M1v2 (Clinical)', 'm1_oof', 'm1_cal'),
    ('Fusion A: Late Mean', 'fusion_mean_oof', 'fusion_mean_cal'),
    ('Fusion E: Bayesian Evidential (BEF)', 'fusion_bef_oof', 'fusion_bef_cal')
]

for name, raw_key, cal_key in eval_models:
    prob_raw = oof_data[raw_key]
    prob_cal = oof_data[cal_key]
    ece_raw = manifest[name]['ece_raw']
    ece_cal = manifest[name]['ece_cal']
    
    frac_raw, mean_raw = calibration_curve(y, prob_raw, n_bins=6)
    frac_cal, mean_cal = calibration_curve(y, prob_cal, n_bins=6)
    
    ax1.plot(mean_raw, frac_raw, 's--', label=f"{name} (ECE={ece_raw:.3f})",
             color=colors[name], linewidth=1.8, markersize=5)
    ax2.plot(mean_cal, frac_cal, 'o-', label=f"{name} (ECE={ece_cal:.3f})",
             color=colors[name], linewidth=2.0, markersize=5)

for ax, title in [(ax1, "Before Platt Calibration (Raw OOF)"), (ax2, "After 5-Fold Platt Scaling")]:
    ax.plot([0, 1], [0, 1], 'k:', label='Perfect Calibration')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives (M1)')
    ax.set_title(title)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.92)
    ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
fig.savefig(FIGS_DIR / "fig4_calibration_hazard.pdf")
fig.savefig(FIGS_DIR / "fig4_calibration_hazard.png")
plt.close(fig)
print("[FIG] Saved fig4_calibration_hazard")

# -------------------------------------------------------------
# FIGURE 5: Comparative Performance Architecture Barplot
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
names = [
    'M2v2 (Genomic)', 'M3v2 (3D Rad)', 'Fusion G (Inter)', 'Fusion F (GBDT)',
    'Fusion D (MLP)', 'Fusion A (Mean)', 'Fusion C (Logit)', 'Fusion B (Rank)',
    'M1v2 (Clinical)', 'Fusion E (BEF)'
]
aurocs = [manifest[n if 'M3' not in n and 'Fusion G' not in n else ('M3v2 (3D Radiomics)' if 'M3' in n else 'Fusion G: Intermediate Representation')]['auroc'] for n in [
    'M2v2 (Genomic)', 'M3v2 (3D Radiomics)', 'Fusion G: Intermediate Representation',
    'Fusion F: GBDT Stacking', 'Fusion D: Bottleneck MLP', 'Fusion A: Late Mean',
    'Fusion C: Logit Stacking', 'Fusion B: Rank Average', 'M1v2 (Clinical)', 'Fusion E: Bayesian Evidential (BEF)'
]]
ci_lows = [manifest[n]['auroc_ci_low'] for n in [
    'M2v2 (Genomic)', 'M3v2 (3D Radiomics)', 'Fusion G: Intermediate Representation',
    'Fusion F: GBDT Stacking', 'Fusion D: Bottleneck MLP', 'Fusion A: Late Mean',
    'Fusion C: Logit Stacking', 'Fusion B: Rank Average', 'M1v2 (Clinical)', 'Fusion E: Bayesian Evidential (BEF)'
]]
ci_highs = [manifest[n]['auroc_ci_high'] for n in [
    'M2v2 (Genomic)', 'M3v2 (3D Radiomics)', 'Fusion G: Intermediate Representation',
    'Fusion F: GBDT Stacking', 'Fusion D: Bottleneck MLP', 'Fusion A: Late Mean',
    'Fusion C: Logit Stacking', 'Fusion B: Rank Average', 'M1v2 (Clinical)', 'Fusion E: Bayesian Evidential (BEF)'
]]

y_pos = np.arange(len(names))
err_low = np.array(aurocs) - np.array(ci_lows)
err_high = np.array(ci_highs) - np.array(aurocs)

bar_colors = ['#aec7e8', '#aec7e8', '#98df8a', '#98df8a', '#98df8a', '#98df8a', '#98df8a', '#98df8a', '#aec7e8', '#d62728']
bars = ax.barh(y_pos, aurocs, xerr=[err_low, err_high], color=bar_colors, capsize=4, alpha=0.85, edgecolor='k', linewidth=0.8)

ax.set_yticks(y_pos)
ax.set_yticklabels(names)
ax.set_xlim([0.45, 1.0])
ax.axvline(0.84, color='red', linestyle='--', linewidth=1.5, label='SOTA Literature Benchmark (Xiao / Mahootiha: 0.840)')
ax.set_xlabel('5-Fold Cross-Validation AUROC (with 95% Bootstrap CI)')
ax.set_title('Architecture Comparison on ccRCC Metastasis Prediction')
ax.legend(loc='lower right')
ax.grid(True, axis='x', linestyle='--', alpha=0.4)

# Annotate values
for bar, auroc in zip(bars, aurocs):
    ax.text(auroc + 0.02, bar.get_y() + bar.get_height()/2, f"{auroc:.4f}", va='center', ha='left', fontsize=9.5, fontweight='bold')

plt.tight_layout()
fig.savefig(FIGS_DIR / "fig5_architecture_comparison.pdf")
fig.savefig(FIGS_DIR / "fig5_architecture_comparison.png")
plt.close(fig)
print("[FIG] Saved fig5_architecture_comparison")
print("\nAll publication figures successfully updated!")
