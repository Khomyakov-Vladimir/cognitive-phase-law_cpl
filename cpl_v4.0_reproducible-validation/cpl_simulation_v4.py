#!/usr/bin/env python3
"""
cpl_simulation_v4.py
CPL 4.0: Cognitive Phase System (reproducible implementation)

VERSION 4.0 - Supersedes Version 3.0 (doi:10.5281/zenodo.17679840)
Author: Vladimir Khomyakov
License: MIT
"""

import numpy as np
from scipy.stats import entropy as scipy_entropy
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import warnings
import os
import json
from datetime import datetime

np.random.seed(42)
warnings.filterwarnings('ignore')

# Version metadata (essential for reproducibility)
__version__ = "4.0.0"
__supersedes__ = "3.0.0 (doi:10.5281/zenodo.17679840)"


class CognitivePhaseSystemV4:
    """
    Self-organizing cognitive phase dynamics (CPL 4.0).

    Parameters
    ----------
    n_projections : int
        Number of cognitive projections (default: 5)
    alpha : float
        Learning rate (default: 0.08)
    gamma : float
        Critical entropy rate threshold (default: 0.1)
    beta_hyst : float
        Hysteresis strength (default: 0.05)
    H_target : float, optional
        Target entropy level in nats
    initial_weights : np.ndarray, optional
        Custom initial weights (overrides Boltzmann init)
    """
    def __init__(self, 
                 n_projections: int = 5, 
                 alpha: float = 0.08,
                 gamma: float = 0.1, 
                 beta_hyst: float = 0.05, 
                 H_target: Optional[float] = None,
                 initial_weights: Optional[np.ndarray] = None):
        self.H_c = np.log(3)  # critical entropy threshold
        self.S_c = 0.7        # critical stability threshold
        self.n_projections = n_projections
        self.alpha = alpha
        self.gamma = gamma
        self.beta_hyst = beta_hyst

        if H_target is None:
            self.H_target = np.random.uniform(0.6, 1.4)
        else:
            self.H_target = H_target

        if initial_weights is not None:
            if len(initial_weights) != n_projections:
                raise ValueError(f"initial_weights must have length {n_projections}")
            self.w = np.array(initial_weights, dtype=float)
            self.w /= (np.sum(self.w) + 1e-12)
        else:
            self.w = self._initialize_weights_boltzmann()

        self.phase_history = []
        self.reorg_counter = 0
        self.last_stable_phase = None
        self.version = __version__
        self.metadata = {
            'version': __version__,
            'supersedes': __supersedes__,
            'created': datetime.now().isoformat(),
            'parameters': {
                'n_projections': n_projections,
                'alpha': alpha,
                'gamma': gamma,
                'beta_hyst': beta_hyst,
                'H_target': self.H_target,
                'H_c': self.H_c,
                'S_c': self.S_c
            }
        }

    def _initialize_weights_boltzmann(self) -> np.ndarray:
        n = self.n_projections
        energy_levels = np.linspace(0, 1, n)
        if self.H_target > 0.1:
            beta_approx = np.clip(n / (2.0 * self.H_target) - 1.0, 0.0, 20.0)
        else:
            beta_approx = 20.0
        w = np.exp(-beta_approx * energy_levels)
        w /= np.sum(w)
        H_actual = -np.sum(w * np.log(w + 1e-12))
        if abs(H_actual - self.H_target) > 0.2:
            def entropy_error(beta):
                w_temp = np.exp(-beta * energy_levels)
                w_temp /= np.sum(w_temp)
                return -np.sum(w_temp * np.log(w_temp + 1e-12)) - self.H_target
            try:
                beta_opt = brentq(entropy_error, 0.01, 20.0)
                w = np.exp(-beta_opt * energy_levels)
                w /= np.sum(w)
            except ValueError:
                pass
        return w

    def sigma(self, w: np.ndarray) -> np.ndarray:
        gain = 6.0
        w_mean = np.mean(w)
        return np.tanh(gain * (w - w_mean))

    def compute_state(self) -> Tuple[float, float]:
        H_obs = -np.sum(self.w * np.log(self.w + 1e-12))
        S = np.max(self.w)
        return H_obs, S

    def dynamics(self, w: np.ndarray, t: float, noise: float = 0.02) -> np.ndarray:
        w_normalized = w / (np.sum(w) + 1e-12)
        self.w = w_normalized
        H_obs, S = self.compute_state()
        entropy_diff = H_obs - self.H_target
        learning_signal = self.alpha * entropy_diff * self.sigma(w_normalized)
        noise_term = noise * np.random.randn(len(w))
        return learning_signal + noise_term

    def classify_with_hysteresis(self, H_obs: float, S: float, entropy_rate: float) -> str:
        dH_dt = entropy_rate
        # Update last stable phase
        if self.phase_history:
            for phase in reversed(self.phase_history):
                if phase != 'Reorganization':
                    self.last_stable_phase = phase
                    break

        if self.last_stable_phase == 'Cognitive_Fragmentation':
            H_c_eff = self.H_c + self.beta_hyst
        elif self.last_stable_phase == 'Cognitive_Coherence':
            H_c_eff = self.H_c - self.beta_hyst
        else:
            H_c_eff = self.H_c

        if abs(dH_dt) >= self.gamma:
            self.reorg_counter = 3
            return "Reorganization"
        if self.reorg_counter > 0:
            self.reorg_counter -= 1
            return "Reorganization"
        if H_obs < H_c_eff and S > self.S_c:
            return "Cognitive_Coherence"
        else:
            return "Cognitive_Fragmentation"

    def simulate(self, t_max: float = 100, dt: float = 0.1, noise_level: float = 0.02) -> Dict:
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

        tau = 20
        alpha_ema = 0.2

        for i, ti in enumerate(t):
            H_obs, S = self.compute_state()
            H_vals[i] = H_obs
            S_vals[i] = S

            if i > tau:
                raw_dH = (H_vals[i] - H_vals[i - tau]) / (tau * dt)
            elif i > 0:
                raw_dH = (H_vals[i] - H_vals[i - 1]) / dt
            else:
                raw_dH = 0.0

            if i == 0:
                dH_dt = raw_dH
            else:
                dH_dt = alpha_ema * raw_dH + (1 - alpha_ema) * dH_vals[i - 1]
            dH_vals[i] = dH_dt

            phase = self.classify_with_hysteresis(H_obs, S, dH_dt)
            phase_vals.append(phase)
            self.phase_history.append(phase)
            weights_history.append(self.w.copy())

            dw = self.dynamics(self.w, ti, noise=noise_level)
            self.w += dw * dt
            self.w = np.clip(self.w, 1e-8, None)
            self.w /= np.sum(self.w)

        return {
            'time': t,
            'entropy': H_vals,
            'stability': S_vals,
            'entropy_rate': dH_vals,
            'phases': phase_vals,
            'weights_history': weights_history,
            'metadata': self.metadata
        }


def plot_phase_diagram(results: Dict, save_path: Optional[str] = None):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    ax = axes[0]
    ax.plot(results['time'], results['entropy'], 'b-', linewidth=1.5, alpha=0.8)
    ax.axhline(np.log(3), color='k', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.set_ylabel('Observer Entropy $H_{\mathrm{obs}}$ (nats)')
    ax.set_title(f"CPL v{results['metadata']['version']}")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(results['time'], results['stability'], 'g-', linewidth=1.5, alpha=0.8)
    ax.axhline(0.7, color='k', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.set_ylabel('Projection Stability $S(t)$')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    phase_colors = {
        'Cognitive_Coherence': 'blue',
        'Cognitive_Fragmentation': 'red',
        'Reorganization': 'orange'
    }
    for i in range(0, len(results['phases']), 10):
        phase = results['phases'][i]
        phase_idx = list(phase_colors.keys()).index(phase)
        ax.scatter(results['time'][i], phase_idx,
                   c=phase_colors[phase], s=20, alpha=0.7, marker='|', linewidth=2)
    ax.set_ylabel('Cognitive Phase')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Coherence', 'Fragmentation', 'Reorganization'])
    ax.set_xlabel('Time (simulation steps)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    plt.close(fig)


def demonstrate_hysteresis(save_path: Optional[str] = None) -> Tuple[Dict, Dict]:
    system_forward = CognitivePhaseSystemV4(H_target=1.4, alpha=0.1, beta_hyst=0.05, gamma=0.10)
    results_forward = system_forward.simulate(t_max=150, dt=0.1, noise_level=0.015)

    system_reverse = CognitivePhaseSystemV4(H_target=0.7, alpha=0.1, beta_hyst=0.05, gamma=0.10)
    results_reverse = system_reverse.simulate(t_max=150, dt=0.1, noise_level=0.015)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(results_forward['entropy'][::10], results_forward['stability'][::10],
            'b-o', alpha=0.6, markersize=3, linewidth=1.5, label='Forward')
    ax.plot(results_reverse['entropy'][::10], results_reverse['stability'][::10],
            'r-s', alpha=0.6, markersize=3, linewidth=1.5, label='Reverse')
    ax.axvline(np.log(3), color='k', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.axhline(0.7, color='k', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Observer Entropy $H_{\mathrm{obs}}$ (nats)')
    ax.set_ylabel('Projection Stability $S(t)$')
    ax.set_title(f'Hysteresis Loop (CPL v{__version__})')
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    plt.close(fig)
    return results_forward, results_reverse


def run_validation_demo():
    os.makedirs("figures", exist_ok=True)

    # Coherence-dominant
    system1 = CognitivePhaseSystemV4(H_target=0.8, alpha=0.08, beta_hyst=0.05, gamma=0.10)
    results1 = system1.simulate(t_max=200, dt=0.05, noise_level=0.01)
    plot_phase_diagram(results1, "figures/cpl_v4_phase_dynamics.pdf")

    # Fragmentation-dominant
    system2 = CognitivePhaseSystemV4(H_target=1.4, alpha=0.08, beta_hyst=0.05, gamma=0.10)
    results2 = system2.simulate(t_max=200, dt=0.05, noise_level=0.01)
    plot_phase_diagram(results2, "figures/cpl_v4_fragmentation.pdf")

    # Hysteresis
    demonstrate_hysteresis("figures/cpl_v4_hysteresis.pdf")

    # Save metadata
    with open("figures/cpl_v4_metadata.json", "w") as f:
        json.dump(results1['metadata'], f, indent=2)

    print("Validation complete.")
    print("Output files:")
    print("  figures/cpl_v4_phase_dynamics.pdf")
    print("  figures/cpl_v4_fragmentation.pdf")
    print("  figures/cpl_v4_hysteresis.pdf")
    print("  figures/cpl_v4_metadata.json")


if __name__ == "__main__":
    run_validation_demo()
