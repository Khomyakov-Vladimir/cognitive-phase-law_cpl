"""
cognitive_phase_law.py
Illustrative but more reliable implementation of computations for the Cognitive Phase Law (CPL).
Includes entropy calculation, stability and phase classification.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import entropy as scipy_entropy
from typing import Tuple, List

def compute_observer_entropy(z: np.ndarray, n_clusters: int = 10) -> Tuple[float, float]:
    """
    Computes observer entropy and stability based on clustering in latent space.
    Handles edge cases to avoid division by zero and log(0).
    
    Parameters:
    z : np.ndarray
        Array of size (n_samples, n_features) with latent representations.
    n_clusters : int, optional
        Number of clusters for K-Means. Default is 10.
    
    Returns:
    entropy_value : float
        Shannon entropy (base 2) of cluster distribution.
    stability : float
        Stability (maximum fraction of samples in one cluster).
    """
    if z.shape[0] < n_clusters:
        # If samples are fewer than clusters, reduce n_clusters
        n_clusters = max(2, z.shape[0] // 2)  # Minimum 2 clusters
    
    # 1. Clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(z)
    labels = kmeans.labels_
    
    # 2. Computing probability distribution
    counts = np.bincount(labels, minlength=n_clusters)
    probabilities = counts / np.sum(counts)  # Normalize
    
    # 3. ENTROPY CALCULATION (reliable, using scipy)
    # scipy.stats.entropy uses base e, convert_to_base2 for bits
    entropy_value_nats = scipy_entropy(probabilities, base=np.e)
    entropy_value_bits = entropy_value_nats / np.log(2)  # Convert to bits
    
    # 4. STABILITY CALCULATION
    stability = np.max(probabilities)
    
    return entropy_value_bits, stability

def classify_cognitive_phase(H_obs: float, S: float, dH_dt: float,
                            H_c: float = np.log(3), S_c: float = 0.7, gamma: float = 0.1) -> str:
    """
    Classifies cognitive phase based on CPL criteria.
    
    Parameters:
    H_obs : float   # Current observer entropy
    S : float       # Current stability
    dH_dt : float   # Rate of entropy change (derivative)
    H_c : float     # Critical entropy threshold
    S_c : float     # Critical stability threshold
    gamma : float   # Critical entropy change rate threshold
    
    Returns:
    phase_label : str
        Phase label: 'Cognitive_Coherence', 'Cognitive_Fragmentation' or 'Reorganization'
    """
    # Criterion 1: Reorganization (has highest priority)
    if abs(dH_dt) >= gamma:
        return "Reorganization"
    
    # Criterion 2: Cognitive Coherence
    elif H_obs < H_c and S > S_c:
        return "Cognitive_Coherence"
    
    # Criterion 3: Cognitive Fragmentation (all other cases)
    else:
        return "Cognitive_Fragmentation"

def calculate_entropy_rate(H_history: List[float], time_step: float = 1.0) -> float:
    """
    Calculates entropy change rate (derivative) based on value history.
    Uses finite difference.
    
    Parameters:
    H_history : list of float
        List of recent entropy values. [H(t), H(t-1), H(t-2), ...]
    time_step : float
        Time step between measurements.
    
    Returns:
    dH_dt : float
        Current estimate of entropy change rate.
    """
    if len(H_history) < 2:
        # Insufficient data to calculate derivative
        return 0.0
    
    # Simplest approximation: (H(t) - H(t-1)) / Δt
    return (H_history[0] - H_history[1]) / time_step

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    # Data simulation: 100 points in 2D space
    # Create 3 clusters for "Coherent" phase
    np.random.seed(42)
    cluster_1 = np.random.normal(loc=[0, 0], scale=0.2, size=(30, 2))
    cluster_2 = np.random.normal(loc=[1, 1], scale=0.2, size=(50, 2))  # Largest cluster
    cluster_3 = np.random.normal(loc=[2, 2], scale=0.2, size=(20, 2))
    z_coherent = np.vstack([cluster_1, cluster_2, cluster_3])
    
    # Create uniform distribution for "Fragmented" phase
    z_fragmented = np.random.uniform(low=0, high=2, size=(100, 2))
    
    print("=== CPL OPERATION SIMULATION ===")
    
    print("\n1. ANALYSIS OF 'COHERENT' STATE:")
    H_obs_coherent, S_coherent = compute_observer_entropy(z_coherent, n_clusters=5)
    print(f"   Entropy (H_obs): {H_obs_coherent:.3f} bits")
    print(f"   Stability (S): {S_coherent:.3f}")
    
    # Simulate history for derivative: stable entropy -> small derivative
    dH_dt_coherent = 0.01
    phase = classify_cognitive_phase(H_obs_coherent, S_coherent, dH_dt_coherent)
    print(f"   Entropy change rate (dH_dt): {dH_dt_coherent:.3f}")
    print(f"   Predicted phase: {phase}")
    
    print("\n2. ANALYSIS OF 'FRAGMENTED' STATE:")
    H_obs_frag, S_frag = compute_observer_entropy(z_fragmented, n_clusters=5)
    print(f"   Entropy (H_obs): {H_obs_frag:.3f} bits")
    print(f"   Stability (S): {S_frag:.3f}")
    
    # Simulate history for derivative: stable entropy -> small derivative
    dH_dt_frag = 0.01
    phase = classify_cognitive_phase(H_obs_frag, S_frag, dH_dt_frag)
    print(f"   Entropy change rate (dH_dt): {dH_dt_frag:.3f}")
    print(f"   Predicted phase: {phase}")
    
    print("\n3. ANALYSIS OF 'REORGANIZATION':")
    # Use 'coherent' data, but drastically change entropy
    H_obs_current = H_obs_coherent
    S_current = S_coherent
    dH_dt_reorg = 0.5  # High rate of change!
    phase = classify_cognitive_phase(H_obs_current, S_current, dH_dt_reorg)
    print(f"   Entropy (H_obs): {H_obs_current:.3f} bits")
    print(f"   Stability (S): {S_current:.3f}")
    print(f"   Entropy change rate (dH_dt): {dH_dt_reorg:.3f}")
    print(f"   Predicted phase: {phase}")