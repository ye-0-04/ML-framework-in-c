# ML-framework-in-c

A lightweight machine learning framework implemented from scratch in C. The project is focused on learning how tensors, dense layers, forward passes, backpropagation, model serialization, and MNIST-style training work under the hood.

## Overview

The framework provides a small tensor library and a simple feed-forward neural network stack. It can create dense networks, train them with backpropagation, save/load model weights, and run MNIST classification experiments using preprocessed binary data.

## Features

- Multi-dimensional tensor allocation with shape, stride, size, and flat data storage.
- Element-wise tensor operations: addition, subtraction, multiplication, and division.
- In-place tensor operations and scalar operations.
- Tensor utilities: fill, random initialization, copy, clone, reshape, transpose, and printing helpers.
- Matrix multiplication for tensors shaped like `[batch, rows, cols]`.
- Reductions: sum, max, min, mean, argmax, and argmin.
- Broadcasting helper functions.
- Activation functions: ReLU, sigmoid, and softmax.
- Loss/helper functions including mean squared error and entropy.
- Dense neural network layers with cached input, output, and gradient tensors.
- Forward pass for feed-forward classification networks.
- Backpropagation with softmax output and ReLU hidden layers.
- Binary model save/load support.
- MNIST preprocessing scripts and debug harnesses.

## Directory Structure

```text
ML-framework-in-c/
|-- include/
|   |-- tensors.h          # Tensor API declarations
|   |-- neural.h           # Neural network API declarations
|   `-- models.h           # Model save/load declarations
|-- src/
|   |-- tensors.c          # Tensor operations and math helpers
|   |-- neural.c           # Dense layers, forward pass, and backprop
|   `-- models.c           # Binary model serialization
|-- preprocess/
|   |-- data_prerocess.py  # Prepare MNIST training arrays
|   |-- to_bin.py          # Convert training arrays to raw binary files
|   `-- fetch_test.py      # Prepare MNIST test binary files
|-- documentation/         # Project guides and API docs
|-- tests/                 # C test experiments
|-- main.c                 # Main training/evaluation experiment file
|-- debug_one_sample.c     # Fast one-sample overfit test
|-- debug_eval.c           # Evaluation sanity checker
`-- README.md
```

## Build

Compile the main program:

```powershell
gcc -O2 main.c src/tensors.c src/neural.c src/models.c -o tes.exe -Iinclude
```

Compile the one-sample debug test:

```powershell
gcc -O2 debug_one_sample.c src/tensors.c src/neural.c src/models.c -o debug_one_sample.exe -Iinclude
```

Compile the evaluation debugger:

```powershell
gcc -O2 debug_eval.c src/tensors.c src/neural.c src/models.c -o debug_eval.exe -Iinclude
```

## Quick Start

Generate MNIST data:

```powershell
python preprocess\data_prerocess.py
python preprocess\to_bin.py
python preprocess\fetch_test.py
```

Run the one-sample learning test:

```powershell
.\debug_one_sample.exe
```

Run the main experiment:

```powershell
.\tes.exe
```

Generated MNIST files and model artifacts are intentionally ignored by Git.

## Documentation

- [Getting Started](documentation/getting-started.md)
- [Tensor API](documentation/tensor-api.md)
- [Neural Network API](documentation/neural-api.md)
- [Model and Data Files](documentation/model-and-data.md)
- [Debugging Guide](documentation/debugging-guide.md)

## License

This project is open for learning and development purposes.
