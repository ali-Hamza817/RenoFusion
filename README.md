# RenoFusion: Calibration Hazards and Evaluation Artifacts in Decision-Level Multimodal Fusion for Metastasis Prediction in Clear Cell Renal Cell Carcinoma

[![Paper PDF](https://img.shields.io/badge/Paper-Elsevier_2--Column_PDF-red.svg)](paper/main.pdf)
[![Overleaf Source](https://img.shields.io/badge/Overleaf-LaTeX_Source_Zip-green.svg)](paper/RenoFusion_LaTeX_Overleaf_Source.zip)
[![Journal](https://img.shields.io/badge/Journal-Journal_of_Biomedical_Informatics-blue.svg)](https://www.sciencedirect.com/journal/journal-of-biomedical-informatics)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official repository for the research paper: **"RenoFusion: Calibration Hazards and Evaluation Artifacts in Decision-Level Multimodal Fusion for Metastasis Prediction in Clear Cell Renal Cell Carcinoma"** by **Ali Hamza** and **Imran Usman** (*National University of Sciences and Technology - NUST, Islamabad, Pakistan*).

---

## 📌 Abstract

Multimodal decision-level fusion integrates clinical, genomic, and radiomic predictors in oncology, yet the sensitivity of information-theoretic fusion rules to base-model miscalibration and class-imbalance rebalancing remains poorly characterized. This study evaluated whether decision-level multimodal fusion improves discrimination for distant metastasis prediction in clear cell renal cell carcinoma (ccRCC), and quantified how probability calibration and Synthetic Minority Over-sampling Technique rebalancing modulate reported fusion performance. Three single-modality base models (clinical zero-shot XGBoost, genomic XGBoost, CT radiomics on ResNet50 deep features) and seven decision-level fusion strategies (simple average, $F_2$-weighted average, stacking, cascade max, Bayesian Evidence Fusion, Dempster--Shafer Theory, Optimal Transport) were benchmarked on $n = 126$ TCGA-KIRC patients (18 M1; 14.29% prevalence), each scored under three probability regimes with hold-out statistics from 5-fold stratified cross-validation and 2,000 paired bootstrap resamples. M3 CT radiomics achieved the highest single-modality discrimination (AUROC 0.7037 [95% CI, 0.5622--0.8277]; recall 0.8889; $F_2$ 0.5298). Uncalibrated information-theoretic fusion produced artifactual AUROC gains up to 0.7505 from log-odds stretching of SMOTE-rebalanced probabilities. After 5-fold Platt cross-calibration, all fusion gains dissolved: Platt $F_2$-weighted (primary endpoint) AUROC 0.6631 ($\Delta = +0.0190, p = 0.779$) and Platt Dempster--Shafer AUROC 0.6831 ($\Delta = +0.0391, p = 0.442$) failed to outperform M3. Residual-error correlation ($\rho \in [0.3842, 0.4619]$), Dempster--Shafer evidence conflict ($\bar{K} = 0.4381$, 54.8% high-conflict patients), and SMOTE ablation (+0.0438 artifactual AUROC boost) confirmed the mechanistic origin. Within this alignment-constrained cohort, CT radiomics is the strongest distant-metastasis signal and decision-level fusion yields no statistically significant incremental discrimination. 5-fold cross-validated Platt scaling of base-model out-of-fold probabilities must precede fusion to prevent artifactual performance reporting.

---

## 🔑 Key Research Findings

1. **CT Radiomics Dominance (M3):**
   - Pre-operative CT radiomics (M3) is the strongest single-modality discriminator (**AUROC 0.7037**, 95% CI [0.5622--0.8277], Brier Skill Score **+0.0515**, $F_2$ **0.5298**).
   - M3 outperforms clinical zero-shot transfer (M1; AUROC 0.5592) and genomic mutation prediction (M2; AUROC 0.6672).

2. **The Calibration--Fusion Hazard:**
   - Uncalibrated information-theoretic fusion rules operating in log-odds space (BEF, DST, Optimal Transport) produce apparent AUROC gains up to **0.7505**.
   - These gains derive entirely from log-odds stretching of SMOTE-rebalanced probabilities rather than genuine cross-modal complementarity.

3. **Dissolution under 5-Fold Platt Cross-Calibration:**
   - Because Platt scaling is strictly rank-preserving per fold ($\Delta \text{AUROC} = 0.00$ per fold), applying Platt scaling to out-of-fold probabilities rescales log-odds while preserving within-fold rank order.
   - Following Platt calibration, all artificial fusion gains vanish: Platt $F_2$-weighted AUROC drops to **0.6631** ($p = 0.779$ vs M3) and Platt Dempster--Shafer AUROC drops to **0.6831** ($p = 0.442$ vs M3). No fusion rule statistically significantly outperforms single-modality CT radiomics.

4. **Mechanistic Decompositions:**
   - **Inter-Modality Error Correlation:** Off-diagonal residual error correlations ($\rho \in [0.3842, 0.4619]$, $p < 0.001$) indicate shared failure modes that prevent variance reduction under ensembling.
   - **Dempster--Shafer Conflict:** Mean cohort conflict $\bar{K} = 0.4381$ (54.8% high-conflict cases) causes DST normalization to amplify small belief instabilities.
   - **SMOTE Probability Stretching:** SMOTE contributes a $+0.0438$ uncalibrated AUROC boost that is fully reversed by Platt pre-calibration.

---

## 📊 Benchmark Results

### 1. Base Model Out-of-Fold Performance ($n = 126$, Uncalibrated)

| Model | AUROC (95% CI) | AUPRC (95% CI) | Recall (95% CI) | Precision (95% CI) | $F_2$ (95% CI) | Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1 Clinical** | 0.5592 [0.4205--0.6965] | 0.1634 [0.1177--0.3130] | 0.9444 [0.8333--1.0000] | 0.1717 [0.1485--0.1935] | 0.4971 [0.4294--0.5455] | 0.3123 |
| **M2 Genomic (5-CV)** | 0.6672 [0.5334--0.7978] | 0.2265 [0.1510--0.4274] | 1.0000 [1.0000--1.0000] | 0.1622 [0.1525--0.1731] | 0.4918 [0.4737--0.5114] | 0.1442 |
| **M3 Imaging (5-CV)** | **0.7037** [0.5622--0.8277] | **0.3701** [0.1894--0.5742] | 0.8889 [0.7222--1.0000] | **0.2025** [0.1647--0.2400] | **0.5298** [0.4333--0.6081] | **0.1711** |

### 2. Decision Fusion Performance Across 7 Strategies and 3 Regimes vs M3 Imaging

| Strategy | Calibration Regime | AUROC (95% CI) | $\Delta$AUROC vs M3 | Paired 95% CI | $p$-value | Significance |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Fusion A (Simple Avg)** | Uncalibrated | 0.7001 [0.5658--0.8246] | $-0.0036$ | [$-0.1657$, $+0.1729$] | 0.947 | n.s. |
| | Platt | 0.6600 [0.5092--0.7968] | $+0.0159$ | [$-0.1245$, $+0.1533$] | 0.817 | n.s. |
| | Isotonic | 0.7245 [0.5985--0.8354] | $+0.0301$ | [$-0.1286$, $+0.1911$] | 0.715 | n.s. |
| **Fusion B ($F_2$-Weighted, Primary)** | Uncalibrated | 0.7094 [0.5761--0.8318] | $+0.0057$ | [$-0.1517$, $+0.1765$] | 0.949 | n.s. |
| | **Platt** | **0.6631 [0.5128--0.8009]** | **$+0.0190$** | **[$-0.1163$, $+0.1517$]** | **0.779** | **n.s.** |
| | Isotonic | 0.7310 [0.6080--0.8349] | $+0.0365$ | [$-0.1137$, $+0.1914$] | 0.636 | n.s. |
| **Fusion C (Stacking)** | Uncalibrated | 0.6548 [0.5010--0.7989] | $-0.0489$ | [$-0.1898$, $+0.0916$] | 0.511 | n.s. |
| | Platt | 0.6420 [0.4912--0.7860] | $-0.0021$ | [$-0.1008$, $+0.0890$] | 0.989 | n.s. |
| | Isotonic | 0.6888 [0.5581--0.8092] | $-0.0057$ | [$-0.1281$, $+0.1067$] | 0.948 | n.s. |
| **Fusion D (Cascade Max)** | Uncalibrated | 0.6515 [0.5082--0.7855] | $-0.0522$ | [$-0.2135$, $+0.1222$] | 0.543 | n.s. |
| | Platt | 0.6173 [0.4670--0.7659] | $-0.0267$ | [$-0.1734$, $+0.1145$] | 0.730 | n.s. |
| | Isotonic | 0.6777 [0.5489--0.8048] | $-0.0167$ | [$-0.1752$, $+0.1495$] | 0.841 | n.s. |
| **Fusion E (BEF)** | Uncalibrated | 0.7438 [0.6280--0.8503] | $+0.0401$ | [$-0.1044$, $+0.1960$] | 0.585 | n.s. |
| | Platt | 0.6595 [0.5072--0.7989] | $+0.0154$ | [$-0.1214$, $+0.1430$] | 0.810 | n.s. |
| | Isotonic | 0.7197 [0.5964--0.8231] | $+0.0252$ | [$-0.1296$, $+0.1795$] | 0.766 | n.s. |
| **Fusion F (DST)** | Uncalibrated | 0.7449 [0.6286--0.8519] | $+0.0412$ | [$-0.0725$, $+0.1734$] | 0.500 | n.s. |
| | Platt | 0.6831 [0.5478--0.8092] | $+0.0391$ | [$-0.0633$, $+0.1446$] | 0.442 | n.s. |
| | Isotonic | 0.7392 [0.6270--0.8436] | $+0.0448$ | [$-0.0623$, $+0.1559$] | 0.415 | n.s. |
| **Fusion G (OT)** | Uncalibrated | 0.7505 [0.6337--0.8534] | $+0.0468$ | [$-0.0890$, $+0.1929$] | 0.495 | n.s. |
| | Platt | 0.6636 [0.5128--0.7999] | $+0.0195$ | [$-0.1075$, $+0.1410$] | 0.756 | n.s. |
| | Isotonic | 0.7197 [0.5967--0.8241] | $+0.0252$ | [$-0.1268$, $+0.1752$] | 0.749 | n.s. |

---

## 📁 Repository Structure

```
.
├── paper/
│   ├── main.pdf                                # Final compiled Elsevier 2-column PDF
│   ├── main.tex                                # Master LaTeX source
│   ├── references.bib                          # BibTeX references
│   ├── cas-dc.cls                              # Elsevier CAS 2-column class
│   ├── cas-sc.cls                              # Elsevier CAS 1-column class
│   ├── cas-common.sty                          # Elsevier style package
│   ├── figures/                                # Publication figures (Calibration, DCA, SHAP)
│   └── sections/                               # Paper section TeX sources
│       ├── 01_introduction.tex
│       ├── 02_literature_review.tex
│       ├── 03_methodology.tex
│       ├── 04_results.tex
│       ├── 05_discussion.tex
│       └── 06_limitations_future_conclusion.tex
├── scripts/
│   ├── produce_final_results.py                # 5-fold CV, 7 fusion rules, 3 regimes, 2000 bootstraps
│   ├── produce_extended_analytics.py           # Error correlation, DST conflict, SMOTE ablation
│   └── generate_paper_figures.py               # Generates Calibration Curves, DCA, and SHAP plots
├── results/
│   ├── FINAL_AUTHORITATIVE_RESEARCH_REPORT.md # Complete factual statistical report
│   ├── FINAL_AUTHORITATIVE_RESULTS_WITH_CI.csv # Complete metrics CSV with 95% CIs
│   └── FINAL_EXTENDED_ANALYTICS.json           # Raw JSON analytics data
├── webapp/                                     # Flask demonstration web backend
├── vercel_frontend/                            # Vercel frontend interface
├── DEPLOYMENT.md                               # Operational deployment guide
└── README.md                                   # This repository documentation
```

---

## ⚡ Quick Start & Reproduction

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/ali-Hamza817/RenoFusion.git
cd RenoFusion

# Install required Python dependencies
pip install numpy pandas scikit-learn xgboost matplotlib seaborn scipy shap
```

### 2. Run Complete Statistical Pipeline & Resampling

To re-run the 5-fold stratified cross-validation, 7 fusion strategies, 3 calibration regimes, and 2,000 paired bootstrap iterations:

```bash
python3 scripts/produce_final_results.py
python3 scripts/produce_extended_analytics.py
```

### 3. Generate Paper Figures

```bash
python3 scripts/generate_paper_figures.py
```

### 4. Compile Elsevier LaTeX Paper

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

---

## 📜 Citation

If you use this codebase, dataset alignment, or calibration protocol in your research, please cite our paper:

```bibtex
@article{hamza2026renofusion,
  title={RenoFusion: Calibration Hazards and Evaluation Artifacts in Decision-Level Multimodal Fusion for Metastasis Prediction in Clear Cell Renal Cell Carcinoma},
  author={Hamza, Ali and Usman, Imran},
  journal={Journal of Biomedical Informatics},
  year={2026},
  url={https://github.com/ali-Hamza817/RenoFusion},
  publisher={Elsevier}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
