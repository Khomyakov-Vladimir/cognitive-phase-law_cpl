# The Law of Cognitive Phases (CPL): Formalizing Observer State Transitions in Subjective Physics (4.0)  

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16907842.svg)](https://doi.org/10.5281/zenodo.16907842)  
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)  
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)  

This repository contains the complete methodological reformulation of the **Cognitive Phase Law (CPL)** framework. Version 4.0 supersedes Version 3.0 (doi:10.5281/zenodo.17679840) and resolves all methodological ambiguities identified during peer review by establishing a clear two-layer architecture separating theoretical invariants from computational conventions.  

## Abstract  

The Cognitive Phase Law (CPL) formalizes transitions between discrete cognitive regimes. Version 4.0 introduces:  

- **Two-layer architecture**: Explicit separation of Theoretical Layer (CPL Law) from Computational Layer (Algorithmic Conventions)  
- **Phase-dependent hysteresis**: State memory mechanism replacing rate-dependent formulation from v3.0  
- **Boltzmann initialization**: Maximum-entropy principle replacing ad-hoc thresholds  
- **Robust entropy rate estimation**: Adaptive windowing (τ=20) + exponential moving average (α_EMA=0.2)  
- **Bifurcated ensemble design**: Documented scientific rationale for critical region exclusion  
- **Bimodal statistics interpretation**: Correct interpretation of high variance as evidence of attractor stability  

These advances establish CPL 4.0 as a methodologically complete and theoretically consistent model of cognitive phase transitions grounded in Subjective Physics.  

## Repository Structure  

```
the_law_of_cognitive_phases_v4.0/
│
├── cpl_simulation_v4.py		# Core simulation framework with documented conventions
├── cpl_validation_v4.py		# Ensemble bifurcation experiment (n=200)
├── environment-cpl_v4.0.yml		# Conda environment configuration file for
├── requirements-cpl_v4.0.txt		# Contains project dependencies
├── LICENSE
├── README.md
│
└── figures/
    ├── cpl_v4_phase_dynamics.pdf	# Figure 2: Coherence-dominant dynamics
    ├── cpl_v4_fragmentation.pdf	# Figure 3: Stable fragmentation regime
    ├── cpl_v4_hysteresis.pdf		# Figure 4: Phase-dependent hysteresis loop
    ├── ensemble_summary_v4.pdf		# Figure 5: Ensemble validation results
    ├── cpl_v4_metadata.json
    ├── correlation_analysis_v4.pdf
    ├── correlation_H_target_v4.pdf	# Figure 6a: H_target vs transitions
    └── correlation_noise_v4.pdf	# Figure 6b: Noise vs transitions

```

## CPL 4.0: Key Improvements Over Version 3.0  

### 1. Two-Layer Architecture  

**Theoretical Layer (CPL Law - invariant)**:  
- Phase definitions: Coherence (H_obs < H_c, S > S_c), Fragmentation (H_obs ≥ H_c, S ≤ S_c), Reorganization (|Ḣ| ≥ γ)  
- Critical thresholds: H_c = ln(3) ≈ 1.0986 nats, S_c = 0.7, γ = 0.1  
- Dynamical equation: dw_i/dt = α(H_obs − H_target)σ(w_i) + η_i(t)  

**Computational Layer (Algorithmic Conventions - implementation)**:  
- Convention 1: Robust entropy rate estimator (τ=20, α_EMA=0.2)  
- Convention 2: Sigma function with neutral equilibrium at w_i = 1/n  
- Convention 3: Phase-dependent hysteresis (state memory)  
- Convention 4: Boltzmann initialization (max-entropy principle)  
- Convention 5: White Gaussian noise (minimal stochastic driver)  
- Convention 6: Deterministic seeding protocol  

### 2. Resolved Hysteresis Formulation  

**Version 3.0 (deprecated)**:  
H_c^eff = H_c ± β dH/dt  

**Version 4.0 (current standard)**:  
H_c^eff(t) = H_c + β_hyst ⋅ 𝕀[Φ_stable(t)]  

where:  
- 𝕀 = +β_hyst if last stable phase was Fragmentation  
- 𝕀 = −β_hyst if last stable phase was Coherence  
- 𝕀 = 0 if transitioning from Reorganization  

**Rationale**: Rate-dependent formulation unreliable with stochastic noise; phase-dependent form corresponds to energetic hysteresis (memory in potential landscape).  

### 3. Bifurcated Ensemble Design  

**Sampling strategy**:  
H_target ∈ [0.40, 0.85] ∪ [1.35, 1.80] nats  

**Critical region [0.85, 1.35] intentionally excluded**.  

**Three scientific objectives**:  
1. Characterize attractor stability far from criticality  
2. Avoid critical slowing (τ_relax ∼ |H − H_c|^(−ν))  
3. Test spontaneous transition hypothesis  

### 4. Bimodal Statistics Interpretation  

**Observed**: Phase probability statistics exhibit σ ≈ μ ≈ 0.5  

**Correct interpretation**: High variance reflects **bimodal distribution** arising from bifurcated sampling:  
- Systems with H_target < H_c: ~100% Coherence occupancy  
- Systems with H_target > H_c: ~100% Fragmentation occupancy  
- Ensemble mean ≈ 0.5 due to symmetric sampling of two distinct attractor basins  

**This confirms stable attractors, NOT instability!** (Individual trajectories remain in ONE attractor)  

## Simulation Results (CPL 4.0)  

### Single-System Demonstrations  

**Coherence-dominant regime** (H_target = 0.8 nats):  
- Phase transitions: 0 over 4000 steps  
- Mean entropy: 0.800 nats (H < H_c: True)  
- Mean stability: 0.724  
- Interpretation: Stable coherent attractor  

**Fragmentation-dominant regime** (H_target = 1.4 nats):  
- Phase transitions: 0  
- Mean entropy: 1.407 nats (H ≥ H_c: True)  
- Mean stability: 0.357  
- Interpretation: Stable fragmented attractor  

### Hysteresis Validation  

Bidirectional bifurcation experiments demonstrate clear hysteresis loops in (H_obs, S) phase space, confirming path-dependent transitions and state memory consistent with thermodynamic first-order transitions.  

### Ensemble Validation (n=200)  

**Phase Transitions**:  
- Mean ± SD: 12.14 ± 38.02  
- Median: 0.0  
- Interpretation: Rare, discrete events → attractor stability  

**Phase Probabilities (Temporal Occupancy)**:  
- Coherence: 0.471 ± 0.490  
- Fragmentation: 0.528 ± 0.491  
- Reorganization: 0.001 ± 0.003  

**Dwell Times**:  
- Coherence: 5462.9 ± 4788.7 steps (median: 3995.2)  
- Fragmentation: 10187.1 ± 4274.2 steps (median: 12000.0)  
- Reorganization: 6.9 ± 2.0 steps (median: 7.0)  
- Interpretation: Long dwell times → deep attractor basins  

### Correlation Analysis  

A dedicated correlation analysis (n=200) confirms that phase transitions are emergent phenomena, not artifacts of parameter tuning:  

- **Noise independence**: Transition count shows negligible correlation with noise level ($|r| = 0.108 < 0.12$), confirming robustness to stochastic fluctuations.  
- **Bifurcation structure**: A moderate correlation with target entropy ($|r| = 0.234$) reflects the intentional bifurcated sampling design—systems initialized below/above $H_c$ occupy distinct attractor basins.  

These results validate that transitions arise from the geometric structure of the CPL dynamics, not from noise-driven randomness.  

Visualizations:  
- `figures/correlation_H_target_v4.pdf` — correlation between $H_{\text{target}}$ and transition count (Figure 6a)  
- `figures/correlation_noise_v4.pdf` — correlation between noise level and transition count (Figure 6b)  
- `figures/correlation_analysis_v4.pdf` — combined view  

## Installation  

```bash
pip install -r requirements-cpl_v4.0.txt
```

## CPL 4.0 Environment  

### Pip requirements (requirements-cpl_v4.0.txt)  

```
matplotlib==3.10.8
numpy==2.3.5
scipy==1.16.3
```

### Conda environment (environment-cpl_v4.0.yml)  

```yaml
name: cpl_v4-env
channels:
  - conda-forge
dependencies:
  - python
  - numpy=2.3.5
  - scipy=1.16.3
  - matplotlib=3.10.8
```

## Usage  

```bash
# Run single-system demonstrations
python cpl_simulation_v4.py

# Run ensemble bifurcation experiment (n=200)
python cpl_validation_v4.py

# Quick test (n=20)
python cpl_validation_v4.py --n 20

# Skip visualization for large ensembles
python cpl_validation_v4.py --n 500 --no-viz
```

## Reproducibility  

All simulations use deterministic seeding:  
- **Global seed**: `np.random.seed(42)` for single-run demonstrations  
- **Per-run seeds**: $s_i = 1000 + i$ for ensemble validation  

Raw data and figures are automatically archived in `validation_results_v4/` and `figures/` directories with version-tagged filenames.  

## Relation to Subjective Physics  

CPL 4.0 remains directly grounded in the foundational principles of Subjective Physics:  

**Ontological Framework** (Kaminsky 2024-2025):  
- Observer-world mapping: W: Subj → Subj  
- Equivalence classes: {Subj, ∼}  
- Structural irreversibility: Non-invertible coarse-graining  

**Minimal Model** (Versions 10-12):  
- Cognitive projections and observer entropy  
- Cognitive Uncertainty Principle (CUP) establishing information-geometric bounds  

**Hysteresis** as manifestation of subject-object asymmetry:  
The hysteresis demonstrated in Version~4.0 simulations is a direct consequence of the subject–object asymmetry in Subjective Physics: the observer's mapping of states is irreversible because indistinguishability classes do not invert under dynamical evolution. This structural irreversibility manifests computationally as phase-dependent critical thresholds.  

## Theoretical Bridge  

CPL 4.0 establishes the triadic hierarchy of Subjective Physics:  

$$\text{CUP (Cognitive Uncertainty Principle)} \Rightarrow \text{CPL (The Law of Cognitive Phases)} \Rightarrow \text{TCH (Thermodynamic Cognitive Homeostasis)}$$  

as a self-organizing framework grounded in nonlinear dynamics and information geometry.  

## Citation  

```bibtex
@misc{khomyakov_2025_16907842,
  author       = {Khomyakov, Vladimir},
  title        = {The Law of Cognitive Phases (CPL): Formalizing 
                  Observer State Transitions in Subjective Physics},
  month        = dec,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {4.0},
  doi          = {10.5281/zenodo.16907842},
  url          = {https://doi.org/10.5281/zenodo.16907842}
}
```

**Supersedes**: Version 3.0 (doi:10.5281/zenodo.17679840)  

## Keywords  

cognitive phase law, subjective physics, observer entropy, cognitive projection, phase transition, cognitive coherence, cognitive fragmentation, reorganization phase, hysteresis, critical slowing, non-equilibrium cognition, reproducible simulation  

## License  

- **Scientific article and associated documentation** (PDF, figures, LaTeX sources):  
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)  
- **Source code and simulation scripts**:  
  [MIT License](https://opensource.org/licenses/MIT)  

