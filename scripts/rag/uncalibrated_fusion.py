"""
uncalibrated_fusion.py
Implements ad-hoc uncalibrated score normalization and rank fusion methods:
1. Min-Max Convex Combination
2. Z-Score (ZMUV) Fusion
3. Reciprocal Rank Fusion (RRF)
4. Borda Count Fusion
"""

import numpy as np

def min_max_normalize(scores_dict):
    """Rescales scores to [0, 1] per modality."""
    vals = np.array(list(scores_dict.values()))
    min_v, max_v = vals.min(), vals.max()
    if max_v == min_v:
        return {k: 0.5 for k in scores_dict}
    return {k: float((v - min_v) / (max_v - min_v)) for k, v in scores_dict.items()}

def z_score_normalize(scores_dict):
    """Standardizes scores to zero-mean unit-variance (ZMUV)."""
    vals = np.array(list(scores_dict.values()))
    mean_v, std_v = vals.mean(), vals.std()
    if std_v == 0:
        return {k: 0.0 for k in scores_dict}
    return {k: float((v - mean_v) / std_v) for k, v in scores_dict.items()}

def fuse_min_max(score_dicts, weights=None):
    """Convex combination of min-max normalized scores."""
    if weights is None:
        weights = [1.0 / len(score_dicts)] * len(score_dicts)
        
    norm_dicts = [min_max_normalize(sd) for sd in score_dicts]
    all_keys = set().union(*[sd.keys() for sd in score_dicts])
    
    fused = {}
    for k in all_keys:
        fused[k] = sum(w * nd.get(k, 0.0) for w, nd in zip(weights, norm_dicts))
    return fused

def fuse_z_score(score_dicts):
    """Average of Z-score standardized scores."""
    norm_dicts = [z_score_normalize(sd) for sd in score_dicts]
    all_keys = set().union(*[sd.keys() for sd in score_dicts])
    fused = {}
    for k in all_keys:
        fused[k] = sum(nd.get(k, 0.0) for nd in norm_dicts) / len(norm_dicts)
    return fused

def fuse_rrf(score_dicts, k=60):
    """Reciprocal Rank Fusion (RRF): score(d) = sum(1 / (k + rank))."""
    fused = {}
    all_keys = set().union(*[sd.keys() for sd in score_dicts])
    for key in all_keys:
        fused[key] = 0.0
        
    for sd in score_dicts:
        ranked = sorted(sd.items(), key=lambda x: x[1], reverse=True)
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            fused[doc_id] += 1.0 / (k + rank)
    return fused

def fuse_borda(score_dicts):
    """Borda Count Rank Fusion."""
    fused = {}
    all_keys = list(set().union(*[sd.keys() for sd in score_dicts]))
    n_docs = len(all_keys)
    for key in all_keys:
        fused[key] = 0.0
        
    for sd in score_dicts:
        ranked = sorted(sd.items(), key=lambda x: x[1], reverse=True)
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            fused[doc_id] += (n_docs - rank)
    return fused
