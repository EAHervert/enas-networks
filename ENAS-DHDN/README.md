# ENAS-DHDN: Controller and Hypermodel

## Overview
An implementation of a hypermodel based on the DHDN (Densely Connected Hierarchical Network) with ENAS (Efficient Neural Architecture Search) for optimizing image denoising architectures.

## Model Architecture
The network architecture consists of several configurable components:

### Core Components
- **DRC Block**: Configurable number of convolutions and selection between 3x3 and 5x5 kernels
- **Resampling Block**: Configurable down- and upsampling techniques
- **Pre-processor**: Multi-channel image processing pipeline to desired number of channels

### Network Structure
- **Encoder**: 
  - Customizable DRC pairs
  - Downsampling options: Max Pooling (default), Average Pooling, 2x2 Conv
- **Bottleneck**
- **Decoder**:
  - Configurable DRC pairs
  - Upsampling options: Pixel Shuffling (default), Bilinear Interpolation, Transpose Conv

## Training Functions
- **TRAINING_NETWORKS**: Functions to Train the hypernetwork and controller
- **TRAINING_FUNCTIONS**: Functions to help with TRAINING_NETWORKS

## Code Structure

```
├── ENAS_DHDN/
|   ├── __init__.py
│   ├── CONTROLLER.py         # Controller outputting architectures
│   ├── DRC.py                # DRC block
│   ├── RESAMPLING.py         # Downsampling and upsampling modules
│   ├── SHARED_DHDN.py        # Graph-based DHDN extension
│   ├── TRAINING_FUNCTIONS.py # Helper functions for TRAINING_NETWORKS.py
│   └── TRAINING_NETWORKS.py  # Functions to train controller and shared network
└── tests/
    ├── test_CONTROLLER.py          # Unit Test
    ├── test_DRC.py                 # Unit Test
    ├── test_REDUCED_CONTROLLER.py  # Unit Test
    ├── test_RESAMPLING.py          # Unit Test
    └── test_SHARED_DHDN.py         # Unit Test
```

## Getting Started
[Documentation and examples coming soon]