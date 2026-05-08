# Tensor API Documentation

## Overview
This is a simple tensor library for machine learning operations in C.

## Data Structures

### tensor
A structure representing a multi-dimensional array.

**Members:**
- `int* shape` - Array storing dimensions (cols, rows, etc.)
- `int* stride` - Array storing stride values for each dimension
- `double* data` - Pointer to the actual data stored as a flat array
- `int dims` - Number of dimensions
- `int size` - Total number of elements

## Functions

### create_tensor
Creates a new tensor with the specified shape and dimensions.

**Parameters:**
- `shape` - Array specifying the size of each dimension
- `dims` - Number of dimensions

**Returns:** Pointer to the created tensor, or NULL on failure

### add_tensor
Performs element-wise addition of two tensors.

**Parameters:**
- `a` - First tensor
- `b` - Second tensor

**Returns:** New tensor with the result, or NULL on failure

### sub
Performs element-wise subtraction of two tensors.

**Parameters:**
- `a` - First tensor
- `b` - Second tensor

**Returns:** New tensor with the result, or NULL on failure

### mul
Performs element-wise multiplication of two tensors.

**Parameters:**
- `a` - First tensor
- `b` - Second tensor

**Returns:** New tensor with the result, or NULL on failure

### div
Performs element-wise division of two tensors.

**Parameters:**
- `a` - First tensor (numerator)
- `b` - Second tensor (denominator)

**Returns:** New tensor with the result, or NULL on failure

### fill
Fills all elements of a tensor with a specified value.

**Parameters:**
- `a` - Tensor to fill
- `b` - Value to fill with

### rand_fill
Fills all elements of a tensor with random double values.

**Parameters:**
- `a` - Tensor to fill

### comp_tensor_size
Compares the size and dimensions of two tensors.

**Parameters:**
- `a` - First tensor
- `b` - Second tensor

**Returns:** 0 if tensors have the same size, 1 otherwise

### copy
Copies data from one tensor to another (both must have same size).

**Parameters:**
- `a` - Destination tensor
- `b` - Source tensor

### reshape
Reshapes a tensor to new dimensions (total size must remain the same).

**Parameters:**
- `a` - Tensor to reshape
- `shape` - New shape array
- `dims` - New number of dimensions

### scal_add
Adds a scalar value to all elements of a tensor (in-place).

**Parameters:**
- `a` - Tensor to modify
- `b` - Scalar value to add

### scal_mul
Multiplies all elements of a tensor by a scalar value (in-place).

**Parameters:**
- `a` - Tensor to modify
- `b` - Scalar value to multiply by

## Usage Example

```c
int shape[] = {2, 3};
tensor* t = create_tensor(shape, 2);
fill(t, 1.0);
rand_fill(t);
reshape(t, 3, 2);
```