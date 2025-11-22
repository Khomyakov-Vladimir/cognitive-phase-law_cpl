#!/usr/bin/env python3
"""
cpl_validation_v3.py
PHYSICALLY CORRECT Ensemble Validation for CPL 3.0
All values are now aligned with the theoretically stable demonstration regime.
Key corrections:
  • alpha: 0.08 (reduced from 0.30) - matches demo stability
  • beta: 0.05 - consistent hysteresis strength (matches demo)
  • gamma: 0.10 - suppresses false reorganizations
  • H_target sampling: Added safety gap around H_c to avoid critical slowing
  • noise: 0.01–0.04 - physiological level
  • reorg_dwell: 3 steps - allows phase stabilization
All entropy units are nats.
"""
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from typing import List, Dict
from cpl_simulation_v3 import CognitivePhaseSystem

np.random.seed(42)

def initialize_weights_for_target(H_target: float, n_proj: int = 5):
    """
    Ensure proper initialization depending on regime:
      • H_target < H_c: coherent (dominant projection)
      • H_target > H_c: fragmented (uniform)
    """
    H_c = np.log(3)
    if H_target < H_c:
        w = np.random.rand(n_proj)
        dominant = np.random.randint(0, n_proj)
        w[:] = 0.05
        w[dominant] = 1.0
        w /= np.sum(w)
    else:
        w = np.ones(n_proj) / n_proj
    return w

def run_single(seed: int, H_target: float, noise: float, out_dir: str):
    np.random.seed(seed)
    init_w = initialize_weights_for_target(H_target)
    # CORRECTED PARAMETERS - aligned with demo stability
    system = CognitivePhaseSystem(
        n_projections=5,
        alpha=0.08,              # matches demo
        beta=0.05,               # ✅ now consistent with demo
        gamma=0.10,              # higher reorg threshold
        H_target=H_target,
        initial_weights=init_w
    )
    results = system.simulate(
        t_max=600,
        dt=0.05,
        noise_level=noise
    )
    fname = os.path.join(out_dir, f"run_seed{seed}_Ht{H_target:.3f}_noise{noise:.3f}.pkl")
    with open(fname, "wb") as f:
        pickle.dump(results, f)
    return results

def ensemble_validation(n=200, out_dir="validation_results"):
    os.makedirs(out_dir, exist_ok=True)
    meta = []
    H_c = np.log(3)
    # CRITICAL FIX: Add safety gap around H_c to avoid critical slowing
    # Low regime: 0.40 - 0.85 (gap: 0.85 to 1.099)
    # High regime: 1.35 - 1.80 (gap: 1.099 to 1.35)
    H_targets_low = np.random.uniform(0.40, 0.85, n // 2)
    H_targets_high = np.random.uniform(1.35, 1.80, n // 2)
    H_targets = np.concatenate([H_targets_low, H_targets_high])
    # Reduced noise to physiological level (0.01-0.04)
    noises = np.random.uniform(0.01, 0.04, n)

    for i in range(n):
        seed = 1000 + i
        Ht = float(H_targets[i])
        noise = float(noises[i])
        print(f"Run {i+1}/{n}: seed={seed}, H_target={Ht:.3f}, noise={noise:.3f}")
        res = run_single(seed, Ht, noise, out_dir)
        phases = res['phases']
        transitions = sum(1 for j in range(1, len(phases)) if phases[j] != phases[j-1])
        frac_coh = phases.count('Cognitive_Coherence') / len(phases)
        frac_frag = phases.count('Cognitive_Fragmentation') / len(phases)
        frac_reorg = phases.count('Reorganization') / len(phases)

        meta.append({
            "seed": seed,
            "H_target": Ht,
            "noise": noise,
            "transitions": transitions,
            "frac_coh": frac_coh,
            "frac_frag": frac_frag,
            "frac_reorg": frac_reorg
        })

    summary_path = os.path.join(out_dir, "ensemble_summary.pkl")
    with open(summary_path, "wb") as f:
        pickle.dump(meta, f)
    return meta

def compute_dwell_times(phases: List[str]) -> Dict[str, float]:
    """Compute mean dwell time for each phase from a list of phase labels."""
    dwell_coh, dwell_frag, dwell_reorg = [], [], []
    if not phases:
        return {"dwell_coh": 0.0, "dwell_frag": 0.0, "dwell_reorg": 0.0}
    current_phase = phases[0]
    count = 1
    for phase in phases[1:]:
        if phase == current_phase:
            count += 1
        else:
            if current_phase == 'Cognitive_Coherence':
                dwell_coh.append(count)
            elif current_phase == 'Cognitive_Fragmentation':
                dwell_frag.append(count)
            elif current_phase == 'Reorganization':
                dwell_reorg.append(count)
            current_phase = phase
            count = 1
    # Final segment
    if current_phase == 'Cognitive_Coherence':
        dwell_coh.append(count)
    elif current_phase == 'Cognitive_Fragmentation':
        dwell_frag.append(count)
    elif current_phase == 'Reorganization':
        dwell_reorg.append(count)
    return {
        "dwell_coh": np.mean(dwell_coh) if dwell_coh else 0.0,
        "dwell_frag": np.mean(dwell_frag) if dwell_frag else 0.0,
        "dwell_reorg": np.mean(dwell_reorg) if dwell_reorg else 0.0
    }

def visualize_ensemble_summary(meta: List[Dict], out_dir: str = "validation_results"):
    """
    Generate publication-quality ensemble summary figure.
    """
    # Extract statistics
    transition_counts = [m['transitions'] for m in meta]
    coherence_probs = [m['frac_coh'] for m in meta]
    fragmentation_probs = [m['frac_frag'] for m in meta]
    reorg_probs = [m['frac_reorg'] for m in meta]
    # Compute dwell times
    dwell_time_coherence = []
    dwell_time_fragmentation = []
    dwell_time_reorg = []
    first_result = None
    for m in meta:
        fname = os.path.join(out_dir, f"run_seed{m['seed']}_Ht{m['H_target']:.3f}_noise{m['noise']:.3f}.pkl")
        if os.path.exists(fname):
            with open(fname, 'rb') as f:
                res = pickle.load(f)
            if first_result is None:
                first_result = res
            phases = res['phases']
            dwell_stats = compute_dwell_times(phases)
            if dwell_stats["dwell_coh"] > 0:
                dwell_time_coherence.append(dwell_stats["dwell_coh"])
            if dwell_stats["dwell_frag"] > 0:
                dwell_time_fragmentation.append(dwell_stats["dwell_frag"])
            if dwell_stats["dwell_reorg"] > 0:
                dwell_time_reorg.append(dwell_stats["dwell_reorg"])
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # Panel 1: Transitions
    ax = axes[0,0]
    ax.hist(transition_counts, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel('Number of Phase Transitions')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Transitions Across Ensemble')
    ax.grid(True, alpha=0.3)
    # Panel 2: Phase probabilities
    ax = axes[0,1]
    phases = ['Coherence', 'Fragmentation', 'Reorganization']
    probs = [np.mean(coherence_probs), np.mean(fragmentation_probs), np.mean(reorg_probs)]
    ax.bar(phases, probs, color=['blue', 'red', 'orange'], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Mean Phase Probability')
    ax.set_title('Temporal Occupancy of Cognitive Phases')
    ax.grid(True, alpha=0.3)
    # Panel 3: Dwell times
    ax = axes[1,0]
    dwell_data, dwell_labels = [], []
    if dwell_time_coherence:
        dwell_data.append(dwell_time_coherence); dwell_labels.append('Coherence')
    if dwell_time_fragmentation:
        dwell_data.append(dwell_time_fragmentation); dwell_labels.append('Fragmentation')
    if dwell_time_reorg:
        dwell_data.append(dwell_time_reorg); dwell_labels.append('Reorganization')
    if dwell_data:
        ax.boxplot(dwell_data, labels=dwell_labels)
        ax.set_ylabel('Dwell Time (simulation steps)')
        ax.set_title('Distribution of Phase Dwell Times')
    ax.grid(True, alpha=0.3)
    # Panel 4: Representative trajectory
    ax = axes[1,1]
    if first_result:
        phase_colors = {'Cognitive_Coherence': 'blue', 'Cognitive_Fragmentation': 'red', 'Reorganization': 'orange'}
        phase_numeric = [list(phase_colors.keys()).index(p) for p in first_result['phases']]
        ax.scatter(first_result['time'], phase_numeric, c=[phase_colors[p] for p in first_result['phases']],
                   s=15, alpha=0.7, marker='|', linewidth=2)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Coherence', 'Fragmentation', 'Reorganization'], rotation=45)
        ax.set_xlabel('Time (simulation steps)')
        ax.set_title('Representative Temporal Evolution')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    save_path = 'figures/ensemble_summary.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Ensemble summary figure saved to '{save_path}'")
    plt.show()

if __name__ == "__main__":
    print("Running CPL 3.0 Physically Correct Validation...")
    print("Parameters aligned with stable demonstration regime\n")
    results = ensemble_validation(n=200, out_dir="validation_results")
    transitions = [m['transitions'] for m in results]
    coh_fracs = [m['frac_coh'] for m in results]
    frag_fracs = [m['frac_frag'] for m in results]
    reorg_fracs = [m['frac_reorg'] for m in results]
    # Dwell times (for console output)
    dwell_coh, dwell_frag, dwell_reorg = [], [], []
    for m in results:
        fname = f"validation_results/run_seed{m['seed']}_Ht{m['H_target']:.3f}_noise{m['noise']:.3f}.pkl"
        with open(fname, 'rb') as f:
            res = pickle.load(f)
        stats = compute_dwell_times(res['phases'])
        if stats['dwell_coh'] > 0:
            dwell_coh.append(stats['dwell_coh'])
        if stats['dwell_frag'] > 0:
            dwell_frag.append(stats['dwell_frag'])
        if stats['dwell_reorg'] > 0:
            dwell_reorg.append(stats['dwell_reorg'])
    print("\n=== ENSEMBLE STATISTICS (PHYSICALLY CORRECT) ===")
    print(f"Mean transitions: {np.mean(transitions):.2f} ± {np.std(transitions):.2f}")
    print(f"Coherence probability: {np.mean(coh_fracs):.3f} ± {np.std(coh_fracs):.3f}")
    print(f"Fragmentation probability: {np.mean(frag_fracs):.3f} ± {np.std(frag_fracs):.3f}")
    print(f"Reorganization probability: {np.mean(reorg_fracs):.3f} ± {np.std(reorg_fracs):.3f}")
    print(f"Dwell time (Coherence): {np.mean(dwell_coh):.1f} ± {np.std(dwell_coh):.1f} steps")
    print(f"Dwell time (Fragmentation): {np.mean(dwell_frag):.1f} ± {np.std(dwell_frag):.1f} steps")
    print(f"Dwell time (Reorganization): {np.mean(dwell_reorg):.1f} ± {np.std(dwell_reorg):.1f} steps")
    print("Results saved to 'validation_results/'.")
    visualize_ensemble_summary(results, out_dir="validation_results")
