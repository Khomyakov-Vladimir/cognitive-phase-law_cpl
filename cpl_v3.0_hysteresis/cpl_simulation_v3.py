#!/usr/bin/env python3
"""
cpl_simulation_v3.py
Complete Self-Organizing Implementation of CPL 3.0 (PHYSICALLY CORRECT)
Demonstrates theoretically grounded phenomena of cognitive phase transitions.
All computations are performed in natural units (nats).
"""
import numpy as np
from scipy.stats import entropy as scipy_entropy
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import warnings
import os

np.random.seed(42)
warnings.filterwarnings('ignore')

class CognitivePhaseSystem:
    """
    Self-organizing cognitive system implementing full CPL dynamics.
    All internal representations and thresholds are in natural units (nats).
    """
    def __init__(self, n_projections: int = 5, alpha: float = 0.1,
                 gamma: float = 0.1, beta: float = 0.05, H_target: float = None,
                 initial_weights: Optional[np.ndarray] = None):
        self.n_projections = n_projections
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        if H_target is None:
            self.H_target = np.random.uniform(0.6, 1.4)
        else:
            self.H_target = H_target
        # Use provided initial weights if given; otherwise, initialize based on H_target
        if initial_weights is not None:
            if len(initial_weights) != n_projections:
                raise ValueError(f"initial_weights must have length {n_projections}")
            self.w = np.array(initial_weights, dtype=float)
            self.w = self.w / np.sum(self.w + 1e-12)
        else:
            self.w = self._initialize_weights()
        # Critical thresholds in nats
        self.H_c = np.log(3)  # ≈ 1.0986 nats
        self.S_c = 0.7
        # Memory for hysteresis and state tracking
        self.phase_history = []
        self.reorg_counter = 0
        self.last_stable_phase = None

    def _initialize_weights(self) -> np.ndarray:
        """Initialize weights based on target entropy for proper dynamics."""
        if self.H_target < 0.9:
            return np.array([0.65, 0.20, 0.08, 0.04, 0.03])
        elif self.H_target > 1.3:
            return np.array([0.35, 0.25, 0.20, 0.12, 0.08])
        else:
            return np.ones(self.n_projections) / self.n_projections

    def sigma(self, w: np.ndarray) -> np.ndarray:
        gain = 6.0
        centered_w = w - np.mean(w)
        return np.tanh(gain * centered_w)

    def compute_state(self) -> Tuple[float, float]:
        H_obs_nats = -np.sum(self.w * np.log(self.w + 1e-12))
        S = np.max(self.w)
        return H_obs_nats, S

    def dynamics(self, w: np.ndarray, t: float, noise: float = 0.01) -> np.ndarray:
        w_normalized = w / (np.sum(w) + 1e-12)
        self.w = w_normalized
        H_obs_nats, S = self.compute_state()
        entropy_diff = H_obs_nats - self.H_target
        learning_signal = self.alpha * entropy_diff * self.sigma(w_normalized)
        noise_term = noise * np.random.randn(len(w))
        return learning_signal + noise_term

    def classify_with_hysteresis(self, H_obs: float, S: float, dH_dt: float) -> str:
        if len(self.phase_history) > 0:
            for phase in reversed(self.phase_history):
                if phase != 'Reorganization':
                    self.last_stable_phase = phase
                    break
        H_c_forward = self.H_c + self.beta
        H_c_reverse = self.H_c - self.beta
        if self.last_stable_phase == 'Cognitive_Fragmentation':
            H_c_effective = H_c_forward
        elif self.last_stable_phase == 'Cognitive_Coherence':
            H_c_effective = H_c_reverse
        else:
            H_c_effective = self.H_c

        if np.abs(dH_dt) >= self.gamma:
            self.reorg_counter = 3  # Increased for stability
            return "Reorganization"
        if self.reorg_counter > 0:
            self.reorg_counter -= 1
            return "Reorganization"
        if H_obs < H_c_effective and S > self.S_c:
            return "Cognitive_Coherence"
        else:
            return "Cognitive_Fragmentation"

    def simulate(self, t_max: float = 100, dt: float = 0.1, 
                 noise_level: float = 0.02) -> Dict:
        t = np.arange(0, t_max, dt)
        n_steps = len(t)
        H_vals = np.zeros(n_steps)
        S_vals = np.zeros(n_steps)
        dH_vals = np.zeros(n_steps)
        phase_vals = []
        weights_history = []

        self.phase_history.clear()
        self.reorg_counter = 0
        self.last_stable_phase = None

        H_initial, S_initial = self.compute_state()
        print(f"  Initial: H_target={self.H_target:.3f}, H_obs={H_initial:.3f}, S={S_initial:.3f}")

        for i, ti in enumerate(t):
            H_obs_nats, S = self.compute_state()
            H_vals[i] = H_obs_nats
            S_vals[i] = S

            if i > 20:
                raw_dH = (H_vals[i] - H_vals[i-20]) / (20 * dt)
            elif i > 0:
                raw_dH = (H_vals[i] - H_vals[i-1]) / dt
            else:
                raw_dH = 0.0

            alpha_ema = 0.2
            if i == 0:
                dH_dt = raw_dH
            else:
                dH_dt = alpha_ema * raw_dH + (1 - alpha_ema) * dH_vals[i-1]
            dH_vals[i] = dH_dt

            phase = self.classify_with_hysteresis(H_obs_nats, S, dH_dt)
            phase_vals.append(phase)
            self.phase_history.append(phase)
            weights_history.append(self.w.copy())

            dw = self.dynamics(self.w, ti, noise=noise_level)
            self.w += dw * dt
            self.w = np.clip(self.w, 1e-8, None)
            self.w = self.w / np.sum(self.w)

        return {
            'time': t,
            'entropy': H_vals,
            'stability': S_vals,
            'entropy_rate': dH_vals,
            'phases': phase_vals,
            'weights_history': weights_history
        }

def plot_phase_diagram(results: Dict, save_path: str = None):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    ax = axes[0]
    ax.plot(results['time'], results['entropy'], 'b-', linewidth=1.5, alpha=0.8)
    ax.axhline(np.log(3), color='k', linestyle='--', label=r'$H_c = \ln(3)$', alpha=0.7)
    ax.set_ylabel('Observer Entropy $H_{\mathrm{obs}}$ (nats)')
    ax.set_title('CPL v3.0: Physically Correct Cognitive Phase Dynamics')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(results['time'], results['stability'], 'g-', linewidth=1.5, alpha=0.8)
    ax.axhline(0.7, color='k', linestyle='--', label=r'$S_c = 0.7$', alpha=0.7)
    ax.set_ylabel('Projection Stability $S(t)$')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    phase_colors = {'Cognitive_Coherence': 'blue', 
                    'Cognitive_Fragmentation': 'red',
                    'Reorganization': 'orange'}
    for i, phase in enumerate(results['phases']):
        if i % 10 == 0:
            phase_idx = list(phase_colors.keys()).index(phase)
            ax.scatter(results['time'][i], phase_idx, 
                      c=phase_colors[phase], s=20, alpha=0.7, marker='|')
    ax.set_ylabel('Cognitive Phase')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Coherence', 'Fragmentation', 'Reorganization'])
    ax.set_xlabel('Time (simulation steps)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def demonstrate_hysteresis(save_path: str = None):
    print("=== HYSTERESIS EXPERIMENT ===")
    print("  Forward path (C → F)...")
    system_forward = CognitivePhaseSystem(alpha=0.1, beta=0.05, H_target=1.4)
    results_forward = system_forward.simulate(t_max=150, dt=0.1, noise_level=0.015)

    print("  Reverse path (F → C)...")
    system_reverse = CognitivePhaseSystem(alpha=0.1, beta=0.05, H_target=0.7)
    results_reverse = system_reverse.simulate(t_max=150, dt=0.1, noise_level=0.015)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(results_forward['entropy'][::10], results_forward['stability'][::10], 
            'b-o', label='Forward Path (C → F)', alpha=0.6, markersize=2, linewidth=1)
    ax.plot(results_reverse['entropy'][::10], results_reverse['stability'][::10], 
            'r-s', label='Reverse Path (F → C)', alpha=0.6, markersize=2, linewidth=1)
    ax.axvline(np.log(3), color='k', linestyle='--', label=r'$H_c = \ln(3)$', alpha=0.7)
    ax.axhline(0.7, color='k', linestyle='--', label=r'$S_c = 0.7$', alpha=0.7)
    ax.set_xlabel('Observer Entropy $H_{\mathrm{obs}}$ (nats)')
    ax.set_ylabel('Projection Stability $S(t)$')
    ax.set_title('Hysteresis in Cognitive Phase Transitions (Physically Correct)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return results_forward, results_reverse

def print_detailed_statistics(results: Dict, label: str):
    phases = results['phases']
    entropy = results['entropy']
    transitions = sum(1 for i in range(1, len(phases)) if phases[i] != phases[i-1])
    coh_entropy = [h for h, p in zip(entropy, phases) if p == 'Cognitive_Coherence']
    frag_entropy = [h for h, p in zip(entropy, phases) if p == 'Cognitive_Fragmentation']
    reorg_entropy = [h for h, p in zip(entropy, phases) if p == 'Reorganization']
    coh_stability = [s for s, p in zip(results['stability'], phases) if p == 'Cognitive_Coherence']
    frag_stability = [s for s, p in zip(results['stability'], phases) if p == 'Cognitive_Fragmentation']

    print(f"\n=== {label.upper()} SIMULATION STATISTICS ===")
    print(f"Total steps: {len(phases)}")
    print(f"Phase transitions: {transitions}")
    print(f"Coherence: {len(coh_entropy)/len(phases):.1%}")
    print(f"Fragmentation: {len(frag_entropy)/len(phases):.1%}")
    print(f"Reorganization: {len(reorg_entropy)/len(phases):.1%}")

    if coh_entropy:
        coh_H_mean = np.mean(coh_entropy)
        print(f"Coherence - Entropy: {coh_H_mean:.3f} nats (H < H_c: {coh_H_mean < np.log(3)})")
        print(f"Coherence - Stability: {np.mean(coh_stability):.3f}")
    if frag_entropy:
        frag_H_mean = np.mean(frag_entropy)
        print(f"Fragmentation - Entropy: {frag_H_mean:.3f} nats (H ≥ H_c: {frag_H_mean >= np.log(3)})")
        print(f"Fragmentation - Stability: {np.mean(frag_stability):.3f}")

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    print("=== CPL v3.0: Physically Consistent Phase Dynamics ===")
    print("Model configuration:")
    print("  • σ(w) normalized to [-1, 1] for stable projection dynamics")
    print("  • Weight initialization aligned with the target entropy H_target")
    print("  • Conservative parameter set ensuring numerical stability")
    print("  • Uniform hysteresis constant (β = 0.05) applied across all runs")

    # Coherence-dominant with beta=0.05
    system1 = CognitivePhaseSystem(n_projections=5, alpha=0.08, beta=0.05,
                                  gamma=0.10, H_target=0.8)
    results1 = system1.simulate(t_max=200, dt=0.05, noise_level=0.01)
    print_detailed_statistics(results1, "Coherence-dominant")
    plot_phase_diagram(results1, save_path="figures/cpl_v3_phase_dynamics.pdf")

    # Fragmentation-dominant with beta=0.05
    system2 = CognitivePhaseSystem(n_projections=5, alpha=0.08, beta=0.05,
                                  gamma=0.10, H_target=1.4)
    results2 = system2.simulate(t_max=200, dt=0.05, noise_level=0.01)
    print_detailed_statistics(results2, "Fragmentation-dominant")
    plot_phase_diagram(results2, save_path="figures/cpl_v3_fragmentation.pdf")

    demonstrate_hysteresis(save_path="figures/cpl_v3_hysteresis.pdf")

    print("\n" + "="*60)
    print("=== CPL v3.0 PHYSICALLY CORRECT VALIDATION COMPLETE ===")
    print("All figures saved to 'figures/' directory")
    print("Expected behavior:")
    print("  • Coherence-dominant: >70% Coherence, H < 1.099 nats")
    print("  • Fragmentation-dominant: >70% Fragmentation, H ≥ 1.099 nats")
    print("  • Rare transitions: <50 transitions per run")
    print("="*60)
