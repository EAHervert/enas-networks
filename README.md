# Computationally Efficient NAS for Image Denoising

## Overview
This repository contains the implementation of a Neural Architecture Search (NAS) approach for optimizing image denoising networks while making improvements to the computational cost associated with the search, as described in our paper:

[![DOI](https://img.shields.io/badge/DOI-10.1109%2FACCESS.2025.3557691-blue)](https://doi.org/10.1109/ACCESS.2025.3557691)
[![Paper License: CC BY 4.0](https://img.shields.io/badge/Paper%20License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](./LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12-ee4c2c)](https://pytorch.org/)

*"Computationally Efficient Neural Architecture Search for Image Denoising"*
Esau A. Hervert Hernandez, Yan Cao, Nasser Kehtarnavaz
**Published in IEEE Access**, vol. 13, pp. 60743–60762, 2025.
DOI: [10.1109/ACCESS.2025.3557691](https://doi.org/10.1109/ACCESS.2025.3557691) · [IEEE Xplore](https://ieeexplore.ieee.org/document/10948435)

The paper is © 2025 The Authors, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The code in this repository is released under the [MIT License](./LICENSE).

This is a continuation of our work in the conference proceedings: 
*"Deep Learning Architecture Search for Real-Time Image Denoising"*\
Esau A. Hervert Hernandez, Yan Cao, Nasser Kehtarnavaz\
Proc. SPIE 12102, Real-Time Image Processing and Deep Learning 2022, 1210205 (27 May 2022);\
https://doi.org/10.1117/12.2620349 

## Project Description
We propose an efficient method to optimize the DHDN (Densely Connected Hierarchical Network) architecture using reinforcement learning-based Neural Architecture Search. Our approach builds upon:
- DHDN (Base architecture) - [IEEE](https://ieeexplore.ieee.org/document/9025693/similar#similar)
- NAS - [arXiv:1611.01578](https://arxiv.org/abs/1611.01578)
- ENAS - [PMLR](https://proceedings.mlr.press/v80/pham18a.html)

## Repository Structure
```
├── Benchmarks/                  # Functions to process and analyze results
├── DHDN-ENAS-Network-Search/    # Code containing the Train_Controller, Train_Shared, and Train_ENAS Scripts
├── DHDN-Sensitivity-Analysis/   # Code containing the compare.py function for the sensitivity analysis 
├── ENAS_DHDN/                   # Definition and usage of Controller and Shared Network (hypernetwork)
└── utilities/                   # Helper functions
```

## Datasets
### SIDD Dataset
- Real-world smartphone images
- Multiple lighting conditions and devices
- Paired ground truth/noisy images
- Minimum resolution: 1080p

### DIV2K Dataset
- High-resolution images (2K+)
- Clean ground truth
- Synthetic noise addition with varying σ

## Installation
```bash
# [TODO] Add installation steps
```

## Usage
```python
# [TODO] Add basic usage example
```

## Requirements
- PyTorch
- research-tools  https://github.com/EAHervert/research-tools
- [TODO] Complete dependency list

## Citation

If you use this code or the associated paper, please cite:

```bibtex
@article{hervert2025enas,
  author  = {Hervert Hernandez, Esau A. and Cao, Yan and Kehtarnavaz, Nasser},
  title   = {Computationally Efficient Neural Architecture Search for Image Denoising},
  journal = {IEEE Access},
  volume  = {13},
  pages   = {60743--60762},
  year    = {2025},
  doi     = {10.1109/ACCESS.2025.3557691}
}
```

A machine-readable [`CITATION.cff`](./CITATION.cff) is also provided at the repo root.

## Contact
Esau A. Hervert Hernandez — esauhervert@utexas.edu

*The paper's IEEE record lists `eah170630@utdallas.edu` (UT Dallas); that account is no longer active.*
