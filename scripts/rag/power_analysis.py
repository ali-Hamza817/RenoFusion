"""
power_analysis.py
Computes statistical power curves and minimum detectable effect sizes (MDE)
for medical RAG retrieval and generation evaluation across cohort sample sizes.
"""

import numpy as np
from scipy import stats

def compute_statistical_power(n_queries_list=[50, 100, 150, 200, 500], alpha=0.05, effect_size_d=0.30):
    """
    Computes statistical power for paired comparison tests given sample sizes and effect size.
    """
    power_results = {}
    for n in n_queries_list:
        # Non-centrality parameter for paired t-test
        ncp = effect_size_d * np.sqrt(n)
        crit_val = stats.t.ppf(1 - alpha / 2, df=n - 1)
        # Power = 1 - beta
        power = 1 - stats.t.cdf(crit_val, df=n - 1, loc=ncp) + stats.t.cdf(-crit_val, df=n - 1, loc=ncp)
        power_results[n] = float(power)
    return power_results

def compute_minimum_detectable_effect(n_queries=150, power=0.80, alpha=0.05):
    """
    Calculates the minimum detectable difference (MDE) in Recall/AUROC at 80% power.
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    mde_d = (z_alpha + z_beta) / np.sqrt(n_queries)
    return float(mde_d)

if __name__ == "__main__":
    powers = compute_statistical_power()
    mde = compute_minimum_detectable_effect(150)
    print(f"Statistical Power at N=150 (d=0.30): {powers[150]:.4f}")
    print(f"Minimum Detectable Effect (Cohen's d) at N=150 (80% power): {mde:.4f}")
