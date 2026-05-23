# ML Framework in C - Documentation

This folder documents the current project structure, tensor API, neural network API, model serialization, MNIST preprocessing, and debugging workflow.

## Documentation Map

- [Tensor API](tensor-api.md): tensor structure, tensor creation, math operations, reductions, activations, and memory rules.
- [Neural Network API](neural-api.md): layers, networks, forward pass, backpropagation, shapes, and training flow.
- [Model and Data Files](model-and-data.md): model save/load format, MNIST `.bin` files, preprocessing scripts, and file expectations.
- [Debugging Guide](debugging-guide.md): fast tests for one-sample overfitting, evaluation sanity checks, and common failure modes.
- [Getting Started](getting-started.md): step-by-step guide for generating data, compiling, training, and evaluating.

## Project Layout

```text
ML-framework-in-c/
|-- main.c                  # Train/evaluate entry point used while experimenting
|-- debug_one_sample.c      # Tiny overfit test for checking whether learning works
|-- debug_eval.c            # Evaluation sanity-check tool for saved models
|-- include/
|   |-- tensors.h           # Tensor API declarations
|   |-- neural.h            # Neural network API declarations
|   `-- models.h            # Model save/load declarations
|-- src/
|   |-- tensors.c           # Tensor operations, activations, losses, helpers
|   |-- neural.c            # Layer/network creation, forward pass, backprop
|   `-- models.c            # Binary model serialization
|-- preprocess/
|   |-- data_prerocess.py   # Downloads/prepares normalized MNIST training arrays
|   |-- to_bin.py           # Converts training `.npy` arrays to raw `.bin`
|   `-- fetch_test.py       # Downloads/prepares normalized MNIST test `.bin` files
`-- documentation/
    |-- README.md
    |-- tensor-api.md
    |-- neural-api.md
    |-- model-and-data.md
    `-- debugging-guide.md
```

## Build Commands

Compile the main experiment program:

```powershell
gcc -O2 main.c src/tensors.c src/neural.c src/models.c -o tes.exe -Iinclude
```

Compile the one-sample learning test:

```powershell
gcc -O2 debug_one_sample.c src/tensors.c src/neural.c src/models.c -o debug_one_sample.exe -Iinclude
```

Compile the evaluation checker:

```powershell
gcc -O2 debug_eval.c src/tensors.c src/neural.c src/models.c -o debug_eval.exe -Iinclude
```

## Current Scope

Implemented:

- Multi-dimensional tensor allocation and shape/stride tracking.
- Element-wise tensor math and in-place tensor math.
- Batched-style matrix multiplication for tensors shaped like `[batch, rows, cols]`.
- ReLU, sigmoid, softmax, MSE, and entropy helpers.
- Simple feed-forward dense networks.
- Backpropagation for softmax output and ReLU hidden layers.
- Binary save/load for trained networks.
- MNIST preprocessing and debugging harnesses.

Important limitation:

- Memory management is manual. Every tensor allocation or clone that owns heap memory must eventually be freed.
