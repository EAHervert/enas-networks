## Training Scripts and Usage

This repository includes several scripts for training and evaluating architectures on a hypernetwork:

### Core Training Scripts
- `TRAIN_SHARED.py`: Pre-trains the shared network parameters while keeping controller fixed
- `TRAIN_CONTROLLER.py`: Optimizes the architecture search policy while keeping shared network fixed  
- `ENAS_DHDN_Search.py`: Performs joint training of both controller and shared network using ENAS algorithm

### Evaluation and Analysis
- `COMPARE_ARCHITECTURES.py`: Evaluates and compares performance of different architectures
- `GENERATE_ARCHITECTURES.py`: Generates sample architectures using trained controller for testing

### Code Structure
```
├── TRAIN_SHARED.py           # Trains shared network only
├── TRAIN_CONTROLLER.py       # Trains controller network only
├── ENAS_DHDN_Search.py       # Joint training of both networks
├── COMPARE_ARCHITECTURES.py  # Architecture comparison tools
├── GENERATE_ARCHITECTURES.py # Architecture generation for testing
└── generate_mat.py           # Deprecated as of December 2024
```

### TODO
- Clean up code and refactor repeated code
- Separate training and evaluating code into distinct modules