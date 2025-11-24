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
 └── cpl_v3.0_hysteresis/
        │
        ├── cpl_core.py                 # Core CPL implementation with entropy computation
        ├── cognitive_phase_law.py      # Extended implementation with examples
		├── cpl_simulation_v3.py        # 
		├── cpl_validation_v3.py        # 
		├── figures/
		│ ├── cpl_v3_phase_dynamics.pdf     # Figure 2: Full entropy–stability time-series with phase classification
		│ ├── cpl_v3_fragmentation.pdf      # Figure 3: Stable fragmentation dynamics with H_obs ≥ ln(3)
		│ ├── cpl_v3_hysteresis.pdf         # Figure 4: Forward/Reverse hysteresis loop in (H_obs, S) phase space
		│ └── ensemble_summary.pdf          # Figure 5: Ensemble statistics across 200 simulation runs
        ├── requirements-cpl_v3.0.txt       # Contains project dependencies specification.
		├── environment-cpl_v3.0.yml        # 
        ├── .zenodo.json                # Configuration file for the Zenodo open-access repository
		├── validation_results/
        │ ├── ensemble_summary.pkl                    # 
        │ └── run_seed1XXX_HtX.XXX_noise0.XXX.pkl     # 
        ├── CITATION.cff                # Citation metadata
        ├── LICENSE                     # MIT License
        └── README.md                   # This file
```

---

## CPL 3.0: Empirical Validation Summary  

These results verify stable attractors and rare threshold-driven transitions, exactly as predicted by CPL.  

## Simulation and Algorithmic Description (CPL v3.0)  

### 1. Simulation Engine (`cpl_simulation_v3.py`)  
The CPL v3.0 simulation implements a **fully self-organizing cognitive system**, where all computations are performed in **natural units (nats)**.  
The model evolves a 5-dimensional projection vector *wᵢ* via:  

```
dwᵢ/dt = α · (Hₒᵦₛ − Hₜₐᵣ) · σ(wᵢ) + noise
```

Key scientific features (#1–#5):  
1. **Physically correct entropy computation**  
   Shannon entropy is computed in nats:  
   Hₒᵦₛ = −∑ᵢ wᵢ ln wᵢ  
2. **Dynamic stability metric**  
   S(t) = maxᵢ wᵢ, representing projection dominance.  
3. **Critical thresholds**:  
   H꜀ = ln 3 ≈ 1.099 nats, S꜀ = 0.7 
4. **Hysteresis-aware classification rule**  
   Includes forward/backward thresholds \( H_c ± β \).  
5. **Rare-event Reorganization phase**  
   Triggered by |dH/dt| ≥ γ or transient instability.  

### 2. Script `cpl_validation_v3.py`  
Provides **ensemble-level verification** (200 runs):  
– transition counts,  
– dwell times,  
– entropy/stability distributions,  
– probability of rare reorganizations.  

### 3. Generated Figures  

**Figure 2 — `cpl_v3_fragmentation.pdf` (#6)**  
Shows a prolonged high-entropy regime where Hₒᵦₛ ≥ ln(3) and stability remains low.  
Demonstrates a *stable fragmentation attractor*.  

**Figure 3 — `cpl_v3_hysteresis.pdf` (#7)**  
Displays bidirectional transitions:  
Forward path (Coherence → Fragmentation) and  
Reverse path (Fragmentation → Coherence).  
The two trajectories do not coincide, confirming **path dependence** and **hysteresis loop structure**.  

**Figure 4 — `cpl_v3_phase_dynamics.pdf` (#8)**  
Full time-series of:  
• entropy Hₒᵦₛ(t),  
• stability S(t),  
• phase labels (Coherence / Fragmentation / Reorganization).  
This demonstrates **critical slowing**, **phase dwell times**, and **rare reorganizations**.  

**Figure 5 — `ensemble_summary.pdf` (#9)**  
Statistical summary across 200 independent runs:  
• distribution of phase durations,  
• transition frequencies,  
• empirical entropy/stability histograms.  
Confirms **quantitative agreement** with theoretical CPL predictions.  

---

### Corrections (applied in Version 3.0)  

In earlier releases (1.0–2.0), the script `cognitive_phase_law.py` computed observer entropy in *bits*,  
while the CPL formulation and the `cpl_core.py` module operated strictly in *nats*.  

Starting from Version 3.0, all entropy calculations are fully unified and performed **exclusively in nats**,  
ensuring complete consistency between the theoretical formulation, the empirical validation pipeline,  
and all reference implementations.  

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

If you use this model or code in your research, please cite the original publication:

```bibtex
@misc{khomyakov_2025_17679840,
  author       = {Khomyakov, Vladimir},
  title        = {The Law of Cognitive Phases (CPL): Formalizing
                   Observer State Transitions in Subjective Physics
                  },
  month        = nov,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {3.0},
  doi          = {10.5281/zenodo.17679840},
  url          = {https://doi.org/10.5281/zenodo.17679840}
}
```

Khomyakov, V. (2025). *The Law of Cognitive Phases (CPL): Formalizing Observer State Transitions in Subjective Physics (3.0)*. Zenodo. DOI: [https://doi.org/10.5281/zenodo.17679840]  

- **Version-specific DOI:** [10.5281/zenodo.17679840](https://doi.org/10.5281/zenodo.17679840)  
- **Concept DOI (latest version):** [10.5281/zenodo.16907842](https://doi.org/10.5281/zenodo.16907842)  
- **Download PDF:** [Direct link to paper on Zenodo](https://zenodo.org/records/17679840/files/the_law_of_cognitive_phases_v3.0.pdf) 

---

## References  

**Theoretical Foundation**: This work is based on the principles of Subjective Physics and builds upon the "Minimal Model of Cognitive Projection v10.0" (DOI: [10.5281/zenodo.16888675](https://doi.org/10.5281/zenodo.16888675)) and "Minimal Model of Cognitive Projection v12.0" (DOI: [10.5281/zenodo.17407408](https://doi.org/10.5281/zenodo.17407408)).  

**Empirical Validation**: The CPL framework aligns with recent findings on entropy fluctuations preceding moments of insight ([Tabatabaeian et al., 2025, PNAS](https://doi.org/10.1073/pnas.2502791122)).  

## Keywords  

subjective physics, cognitive projection, observer entropy, cognitive phase law, phase transition, entropy dynamics, hysteresis  

## License  

MIT License. See `LICENSE` file for details.  

