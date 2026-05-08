# ML Framework in C - Documentation

## Project Overview
A lightweight tensor library for machine learning operations written in C.

## File Structure

```
ML-framework-in-c/
├── main.c              # Main entry point (placeholder)
├── include/
│   └── tensors.h       # Header file with function declarations
├── src/
│   └── tensors.c       # Implementation of tensor operations
└── documentation/
    ├── README.md       # This file
    └── tensor-api.md   # Detailed API documentation
```

## Features

- **Tensor Creation**: Create multi-dimensional tensors with custom shapes
- **Element-wise Operations**: Addition, subtraction, multiplication, division
- **Scalar Operations**: Add or multiply tensor by a scalar value
- **Utilities**: Fill tensor with value or random numbers, copy, reshape
- **Validation**: Size and dimension checking for tensor operations

## Building

Compile with a C compiler:
```bash
gcc -o main main.c src/tensors.c -Iinclude
```

## Quick Start

```c
#include "tensors.h"

// Create a 2x3 tensor
int shape[] = {2, 3};
tensor* t = create_tensor(shape, 2);

// Fill with zeros
fill(t, 0.0);

// Fill with random values
rand_fill(t);

// Reshape to 3x2
reshape(t, (int[]){3, 2}, 2);
```

## Notes

- All element-wise operations create new tensors (not in-place)
- Scalar operations modify tensors in-place
- Division by zero is not checked - user must ensure valid inputs
- Memory management is manual - users are responsible for freeing tensors