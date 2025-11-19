#!/usr/bin/env python3
"""
# scripts/cpl_core.py

"The Law of Cognitive Phases (CPL): Formalizing Observer State Transitions in Subjective Physics"

Complete implementation of the core computational functions for the Cognitive Phase Law (CPL).
Includes entropy calculation, stability estimation, and phase classification.

Author: Vladimir Khomyakov  
License: MIT  
Repository: https://github.com/Khomyakov-Vladimir/cognitive-phase-law_cpl 
Citation: DOI:10.5281/zenodo.16907842  
"""

import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import entropy as scipy_entropy
from typing import Tuple

def compute_observer_entropy(z: np.ndarray, n_clusters: int = 10) -> Tuple[float, float]:
    """
    Computes observer entropy and stability based on clustering in a latent space.
    Handles edge cases to avoid division by zero and log(0).

    Parameters:
    z : np.ndarray
        Array of shape (n_samples, n_features) containing latent representations.
    n_clusters : int, optional
        Number of clusters for K-Means. Default is 10.

    Returns:
    entropy_value : float
        Shannon entropy (in nats) of the cluster distribution.
    stability : float
        Stability (maximum proportion of samples in a single cluster).
    """

    n_samples = z.shape[0]
    if n_samples < n_clusters:
        n_clusters = max(2, n_samples // 2)

    # 1. Cluster the latent vectors using K-Means
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(z)
    labels = kmeans.labels_

    # 2. Compute the probability distribution over clusters
    counts = np.bincount(labels, minlength=n_clusters)
    probabilities = counts / np.sum(counts)

    # 3. CALCULATE ENTROPY (in nats)
    entropy_value = scipy_entropy(probabilities, base=np.e)

    # 4. CALCULATE STABILITY
    stability = np.max(probabilities)

    return entropy_value, stability

def calculate_entropy_rate(H_history: list, time_step: float = 1.0) -> float:
    """
    Calculates the rate of change of entropy (derivative) based on a history of values.
    Uses a finite difference approximation.

    Parameters:
    H_history : list of float
        List of recent entropy values. [H(t), H(t-1), H(t-2), ...]
    time_step : float
        Time step between measurements.

    Returns:
    dH_dt : float
        Current estimate of the entropy time derivative.
    """
    if len(H_history) < 2:
        return 0.0
    return (H_history[0] - H_history[1]) / time_step

def classify_cognitive_phase(H_obs: float, S: float, dH_dt: float,
                            H_c: float = np.log(3), S_c: float = 0.7, gamma: float = 0.1) -> str:
    """
    Classifies the cognitive phase based on CPL criteria.

    Parameters:
    H_obs : float   # Current observer entropy (in nats)
    S : float       # Current stability
    dH_dt : float   # Entropy time derivative
    H_c : float     # Critical entropy threshold (default ≈ ln(3))
    S_c : float     # Critical stability threshold
    gamma : float   # Critical entropy rate threshold

    Returns:
    phase_label : str
        Phase label: 'Cognitive_Coherence', 'Cognitive_Fragmentation', or 'Reorganization'
    """
    # Criteria 1: Reorganization (highest priority)
    if abs(dH_dt) >= gamma:
        return "Reorganization"
    # Criteria 2: Cognitive Coherence
    elif H_obs < H_c and S > S_c:
        return "Cognitive_Coherence"
    # Criteria 3: Cognitive Fragmentation (all other cases)
    else:
        return "Cognitive_Fragmentation"

# Note: Empirical simulation and threshold calibration code
# should be implemented in a separate experimental script.
# if __name__ == "__main__":
#     ... (commented out or removed) ...