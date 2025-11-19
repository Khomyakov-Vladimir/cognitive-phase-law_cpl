# The Law of Cognitive Phases (CPL): Formalizing Observer State Transitions in Subjective Physics  

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16907843.svg)](https://doi.org/10.5281/zenodo.16907843)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)  

This repository is the official companion code for the paper:  

> **Vladimir Khomyakov (2025).** _The Law of Cognitive Phases (CPL): Formalizing Observer State Transitions in Subjective Physics (1.0)._ Zenodo. https://doi.org/10.5281/zenodo.16907843  

It provides a complete implementation of the Cognitive Phase Law computational framework described in the paper, enabling reproducible simulations and analysis of cognitive phase transitions.  

## Abstract  

This work introduces the **Cognitive Phase Law (CPL)**, a formal principle describing the dynamics of cognitive state transitions in observers. Building on the Minimal Model of Subjective Physics, CPL characterizes discrete cognitive phases and the conditions under which transitions occur.  

The law predicts **hysteresis effects** and **critical slowing** during cognitive transitions, empirically testable via neurodynamics (EEG, fMRI) and behavioral paradigms. CPL establishes cognition as a **non-equilibrium system** with quantifiable phase-space topology, providing a predictive framework for cognitive projections and observer entropy.  

## Repository Structure  

```
cognitive-phase-law_cpl/
 └── cpl_v1.0_initial/
           │
           ├── cpl_core.py                 # Core CPL implementation with entropy computation
           ├── cognitive_phase_law.py      # Extended implementation with examples
           ├── requirements.txt            # Contains project dependencies specification.
           ├── .zenodo.json                # Configuration file for the Zenodo open-access repository
           ├── CITATION.cff                # Citation metadata
           ├── LICENSE                     # MIT License
           └── README.md                   # This file
```

## Installation  

To run the CPL simulations, you need Python ≥ 3.9. The required libraries can be installed via pip.  

**Create a virtual environment (recommended):**  

```bash
python -m venv cpl-env
# On macOS/Linux:
source cpl-env/bin/activate
# On Windows (PowerShell):
cpl-env\Scripts\activate
```

**Install dependencies:**  

```bash
pip install -r requirements.txt
```

Alternatively, you can install them manually:  

```bash
pip install numpy scipy scikit-learn matplotlib
```

## Usage  

The repository contains two main implementation files:  

1. **cpl_core.py** — Core functions for CPL computation:  
   - `compute_observer_entropy()`: Calculates entropy and stability from latent representations  
   - `calculate_entropy_rate()`: Computes the rate of entropy change  
   - `classify_cognitive_phase()`: Classifies cognitive phases based on CPL criteria  

2. **cognitive_phase_law.py** — Extended implementation with examples:  
   - Includes simulation examples for different cognitive phases  
   - Demonstrates phase classification with realistic data  

## Running the examples  

```bash
# Run the comprehensive example with simulated data
python cognitive_phase_law.py
```

This will generate output showing:  

- Entropy and stability calculations for coherent and fragmented states  
- Phase classification based on CPL criteria  
- Example of reorganization phase detection  

## Expected Results & Outputs  

Running the code will demonstrate:  

**Cognitive Phase Transitions**: The system transitions between coherence, fragmentation, and reorganization phases based on entropy dynamics.  

**Entropy Dynamics**: Observer entropy (H_obs) fluctuates according to cognitive state changes, with critical thresholds triggering phase transitions.  

**Stability Metrics**: Stability (S) measurements show dominance patterns in cognitive projections.  

**Phase Classification**: Automatic classification of cognitive states based on CPL criteria.  

## Citation  

If you use this model or code in your research, please cite the original publication:  

```bibtex
@misc{khomyakov_vladimir_2025_16907843,
  author       = {Khomyakov, Vladimir},
  title        = {The Law of Cognitive Phases (CPL): Formalizing Observer State Transitions in Subjective Physics},
  month        = aug,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.16907843},
  url          = {https://doi.org/10.5281/zenodo.16907843}
}
```

- **Version-specific DOI:** [10.5281/zenodo.16907843](https://doi.org/10.5281/zenodo.16907843)  
- **Concept DOI (latest version):** [10.5281/zenodo.16907842](https://doi.org/10.5281/zenodo.16907842)  
- **Download PDF:** [Direct link to paper on Zenodo](https://zenodo.org/records/16907843/files/the_law_of_cognitive_phases.pdf)  

## References  

**Theoretical Foundation**: This work is based on the principles of Subjective Physics and builds upon the "Minimal Model of Cognitive Projection" (DOI: [10.5281/zenodo.16888675](https://doi.org/10.5281/zenodo.16888675)).  

## Keywords  

subjective physics, cognitive projection, observer entropy, cognitive phase law, phase transition, entropy dynamics, neural synchronization, collective intelligence, information theory, simulation framework  

## License  

This project is licensed under the MIT License. See the LICENSE file for details.  

## Acknowledgements  

This work builds upon the hypothesis of **Subjective Physics** formulated by Alexander Kaminsky, and extends the minimal model of cognitive projection. Special thanks to the researchers whose empirical findings have helped validate the CPL framework.  
