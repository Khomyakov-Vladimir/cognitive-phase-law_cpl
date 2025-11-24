# The Law of Cognitive Phases (CPL): Formalizing Observer State Transitions in Subjective Physics (3.0)  

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17679840.svg)](https://doi.org/10.5281/zenodo.17679840)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

This repository contains the full implementation of the **Cognitive Phase Law (CPL)** framework, now extended with the validated **3.0 empirical and hysteresis results**. Version 3.0 integrates all theoretical material from 2.0 and supplements it with full numerical verification, including hysteresis loops, rare reorganization events, stability statistics, and ensemble-level validation in natural units (nats).  

## Abstract  

The Cognitive Phase Law (CPL) formalizes transitions between discrete cognitive regimes. Version 3.0 introduces a complete numerical validation of the law, demonstrating:  

- hysteresis and path dependence,  
- critical slowing,  
- rare state reorganizations,  
- long dwell times in stable attractors,  
- full consistency of entropy/stability dynamics in natural units.  

These results confirm the predictive and physically consistent nature of CPL as part of Subjective Physics.  

## Repository Structure  

```
cognitive-phase-law_cpl/
 │     ├── LICENSE                     # MIT License
 │     └── README.md                   # This file
 │
 ├── cpl_v1.0_initial/
 │     │
 │     ├── cpl_core.py                 # Core CPL implementation with entropy computation
 │     ├── cognitive_phase_law.py      # Extended implementation with examples
 │     ├── requirements.txt            # Contains project dependencies specification.
 │     ├── .zenodo.json                # Configuration file for the Zenodo open-access repository
 │     ├── CITATION.cff                # Citation metadata
 │     ├── LICENSE                     # MIT License
 │     └── README.md                   # This file
 │
 ├── cpl_v2.0_theory-only/
 │		│
 │		├── cpl_core.py                 # Core CPL implementation with entropy computation
 │		├── cognitive_phase_law.py      # Extended implementation with examples
 │		├── requirements.txt            # Contains project dependencies specification.
 │		├── .zenodo.json                # Configuration file for the Zenodo open-access repository
 │		├── CITATION.cff                # Citation metadata
 │		├── LICENSE                     # MIT License
 │		└── README.md                   # This file
 │
 └── cpl_v3.0_hysteresis/
		│
		├── cpl_core.py					# Core CPL implementation with entropy computation
		├── cognitive_phase_law.py		# Extended implementation with examples
		├── cpl_simulation_v3.py		# Simulation engine with hysteresis
		├── cpl_validation_v3.py		# Ensemble validation scripts
		├── requirements-cpl_v3.0.txt		# Contains project dependencies specification.
		├── environment-cpl_v3.0.yml		# Conda environment configuration file for version 3.0
		├── .zenodo.json				# Configuration file for the Zenodo open-access repository
		├── CITATION.cff				# Citation metadata
		├── LICENSE						# MIT License
		├── README.md					# This file
		│
		├── figures/
		│		├── cpl_v3_phase_dynamics.pdf	# Figure 2: Full entropy–stability time-series with phase classification
		│		├── cpl_v3_fragmentation.pdf	# Figure 3: Stable fragmentation dynamics with H_obs ≥ ln(3)
		│		├── cpl_v3_hysteresis.pdf		# Figure 4: Forward/Reverse hysteresis loop in (H_obs, S) phase space
		│		└── ensemble_summary.pdf		# Figure 5: Ensemble statistics across 200 simulation runs
		│
		└── validation_results/					# Directory for storing validation outputs
				├── ensemble_summary.pkl					# Aggregated statistics from 200 simulation runs
				└── run_seed1XXX_HtX.XXX_noise0.XXX.pkl		# Individual run data files
```

## CPL 3.0: Empirical Validation Summary  

The simulation implements a self-organizing 5-dimensional cognitive projection vector \(w_i\) governed by the dynamical equation:  

    dw_i/dt = α · (H_obs − H_target) · σ(w_i)  

All entropy values and computations are performed in **natural units (nats)** to maintain physical consistency. Phase classification follows the CPL criteria using critical thresholds \(H_c = \ln(3) \approx 1.099\) and \(S_c = 0.7\), with hysteresis-aware path dependence as detailed in the accompanying paper.  

> **Note on unit consistency**: In CPL versions 1.0–2.0, the illustrative script `cognitive_phase_law.py` used entropy in *bits*, which introduced a unit mismatch with the theoretical thresholds (e.g., `H_c = ln(3)`) defined in *natural units (nats)*. The core library `cpl_core.py`—included in the paper and used for all validation experiments—has always correctly used *nats*. Starting from CPL 3.0, the `cognitive_phase_law.py` script has been aligned with the formalism and now also uses *nats* exclusively.  

### Simulation Results  

- **Coherence-dominant run**:  
  49 transitions over 4000 steps  
  H_coh = 0.801 nats  
  H_frag (metastable) = 0.835 nats  

- **Fragmentation-dominant run**:  
  0 transitions  
  H_frag = 1.401 nats (stable fragmentation)  

### Hysteresis  

Bidirectional tests (Coherence→Fragmentation→Coherence) confirm a clear hysteresis loop and state memory. The (H_obs, S) trajectories differ in forward and reverse directions.  

### Ensemble Validation (n=200)  

- Mean transitions: 5.13 ± 19.94  
- Reorganization probability: 0.001 ± 0.003  
- Coherence probability ≈ Fragmentation probability  
- Dwell times:  
  - Coherence: ~6169 steps  
  - Fragmentation: ~11434 steps  
  - Reorganization: ~11 steps  

These results verify stable attractors and rare threshold-driven transitions, exactly as predicted by CPL.  

## Installation  

pip install -r requirements-cpl_v3.0.txt  

## CPL 3.0 Environment  

### Pip requirements (requirements-cpl_v3.0.txt)  

```
matplotlib==3.10.8
numpy==2.3.5
scipy==1.16.3
```

### Conda environment (environment-cpl_v3.0.yml)  

```
name: cpl_v3-env
channels:
  - conda-forge
dependencies:
  - python
  - numpy=2.3.5
  - scipy=1.16.3
  - matplotlib=3.10.8
```

## Usage  

```
python cpl_simulation_v3.py
python cpl_validation_v3.py
```

## Citation  

Version 3.0 DOI will be added upon Zenodo release.  

## Keywords  

subjective physics, cognitive projection, observer entropy, cognitive phase law, phase transition, entropy dynamics, hysteresis  

## License  

MIT License. See `LICENSE` file for details.  

Note: The manuscript (published on Zenodo, DOI: [10.5281/zenodo.17679840](https://doi.org/10.5281/zenodo.17679840)) is licensed under CC BY 4.0; the accompanying source code in this repository is licensed separately under the MIT License.  

## Acknowledgements  

This work builds upon the hypothesis of **Subjective Physics** formulated by Alexander Kaminsky, and extends the minimal model of cognitive projection. Special thanks to the researchers whose empirical findings have helped validate the CPL framework.  
