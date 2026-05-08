# ML-framework-in-c

A lightweight machine learning framework implemented from scratch in C, designed for educational purposes and performance-critical applications.

## Overview

This project provides a foundational ML framework built entirely in C, offering essential tensor operations needed for building neural networks and machine learning algorithms. The framework emphasizes simplicity, efficiency, and a deep understanding of underlying concepts.

## Features

### Implemented
- **Tensor Operations**: Create and manipulate multi-dimensional tensors
- **Element-wise Operations**: Addition, subtraction, multiplication, division
- **Scalar Operations**: In-place scalar addition and multiplication
- **Utilities**: Random initialization, tensor filling, copying, and reshaping

### In Progress / Planned
- Matrix multiplication
- Broadcasting support
- Transpose operations
- Reduction operations (sum, max, min, mean, argmax)
- Activation functions (ReLU, Sigmoid, Softmax)
- Loss functions (MSE, Cross-Entropy)
- Backpropagation

## Project Status

⚠️ **This project is currently under active development and is not yet complete.**

The core tensor data structure and basic operations are implemented. Many advanced features are planned but not yet available. Contributions and feedback are welcome.

## Directory Structure

```
ML-framework-in-c/
├── include/
│   └── tensors.h       # Tensor API header file
├── src/
│   └── tensors.c       # Tensor implementation
├── documentation/
│   ├── README.md
│   └── tensor-api.md   # API documentation
├── main.c              # Example usage
├── future_features.txt # Planned features list
└── README.md           # This file
```

## Building

To compile the project, ensure you have a C compiler installed:

```bash
gcc -o program main.c src/tensors.c -Iinclude
```

## Usage

```c
#include "tensors.h"

int main() {
    int shape[] = {2, 3};
    tensor* a = create_tensor(shape, 2);
    tensor* b = create_tensor(shape, 2);

    rand_fill(a);
    rand_fill(b);

    tensor* c = add_tensor(a, b);

    // ... use tensors

    return 0;
}
```

## Documentation

For detailed API documentation, see `documentation/tensor-api.md`.

## License

This project is open for learning and development purposes.

## Feedback

This project is a personal learning endeavor. You are free to view and experiment with the code. If you have suggestions or ideas, feel free to share them.