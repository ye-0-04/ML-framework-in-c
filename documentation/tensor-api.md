# Tensor API Documentation

The tensor module is the base of the framework. It stores data in one flat `double*` array and uses `shape` and `stride` arrays to interpret that data as a multi-dimensional tensor.

## Data Structure

```c
typedef struct tensor {
    int *shape;
    int *stride;
    double *data;
    int dims;
    int size;
} tensor;
```

- `shape`: array of dimension sizes. For matrix-style tensors in this project, shapes commonly look like `[1, rows, cols]`.
- `stride`: array used to convert multi-dimensional coordinates into flat indexes.
- `data`: contiguous heap array holding all tensor values.
- `dims`: number of dimensions in `shape` and `stride`.
- `size`: total element count, equal to the product of all shape values.

## Creation and Initialization

### `tensor* create_tensor(int shape[], int dims)`

Allocates a tensor wrapper, shape array, stride array, and data array. Data is initialized to `0.0`.

Example:

```c
int shape[] = {1, 784, 1};
tensor *input = create_tensor(shape, 3);
```

The caller owns the returned tensor wrapper. Some project code copies the struct value and frees only the wrapper:

```c
tensor *tmp = create_tensor(shape, 3);
tensor owned_tensor = *tmp;
free(tmp);
```

In that pattern, `owned_tensor` still owns `shape`, `stride`, and `data`.

### `void init_empty_tensor(tensor *a)`

Sets all pointers to `NULL` and numeric fields to zero. Use this before calling `clone()` into a stack tensor.

## Memory Management

### `void free_tensor(tensor *a)`

Frees `shape`, `stride`, and `data`.

Important notes:

- `free_tensor()` does not free the tensor struct itself.
- If the tensor struct was allocated with `malloc`, call `free_tensor(t)` and then `free(t)`.
- If the tensor is a stack variable, call only `free_tensor(&t)`.

## Basic Accessors

- `const double get_tensor_val(tensor *a, int index)`: returns `data[index]`.
- `void set_tensor_val(tensor *a, double val, int index)`: writes to `data[index]`.
- `const int get_tensor_size(tensor *a)`: returns `size`.
- `const int get_tensor_dims(tensor *a)`: returns `dims`.
- `const int* get_tensor_shape(tensor *a)`: returns `shape`.
- `const int* get_tensor_stride(tensor *a)`: returns `stride`.

Indexes are flat indexes, not multi-dimensional coordinate arrays.

## Shape Utilities

### `int comp_tensor_size(tensor *a, tensor *b)`

Returns `0` when tensors have the same dimension count and matching shape values. Returns `1` on mismatch.

### `int reshape(tensor *a, int shape[], int dims)`

Changes tensor metadata to a new shape when the total size is unchanged.

Important caution:

- The current implementation assigns `a->shape = shape`, so the caller must ensure that the new shape storage remains valid.
- Prefer passing a persistent array, not a temporary array that disappears immediately.

### `void transpose(tensor *a, int swap_axes[])`

Swaps two axes by exchanging shape and stride entries. It does not rearrange the underlying data.

Example:

```c
int axes[] = {2, 1};
transpose(&matrix, axes);
```

## Element-Wise Operations

These functions allocate and return a new tensor:

- `tensor* add_tensor(tensor *a, tensor *b, bool con)`
- `tensor* tensor_sub(tensor *a, tensor *b)`
- `tensor* tensor_mul(tensor *a, tensor *b)`
- `tensor* tensor_div(tensor *a, tensor *b)`

`add_tensor(..., con)` has an extra ownership flag:

- `con == false`: leaves inputs alone.
- `con == true`: frees input tensor internals after creating the result.

## In-Place Tensor Operations

These functions modify the first tensor directly:

- `void scal_tensor_add(tensor *a, tensor *b)`: `a[i] += b[i]`
- `void scal_tensor_sub(tensor *a, tensor *b)`: `a[i] -= b[i]`
- `void scal_tensor_mul(tensor *a, tensor *b)`: `a[i] *= b[i]`

Both tensors must have identical shapes.

## Scalar Operations

- `void scal_add(tensor *a, double b)`: adds scalar to every element.
- `void scal_mul(tensor *a, double b)`: multiplies every element by scalar.
- `void scal_pow(tensor *a, int b)`: raises each element to a power.

## Fill and Copy

### `void fill(tensor *a, double b)`

Sets every element to `b`.

### `void rand_fill(tensor *a)`

Fills with small random values centered around zero. This is useful for neural network weights because all-positive large initialization can make training unstable.

### `int copy(tensor *a, tensor *b)`

Copies data from `b` into `a` when both tensors have the same shape.

### `void clone(tensor *a, tensor *b)`

Deep-copies tensor metadata and data from `b` into `a`. If `a` already owns memory, that memory is freed first.

Use `init_empty_tensor(&a)` before the first clone into a stack tensor.

## Matrix Multiplication

### `tensor* tensor_matmul(tensor *a, tensor *b)`

Performs matrix multiplication using the last two dimensions as matrix rows and columns.

Expected shape pattern:

```text
a: [batch, M, K]
b: [batch, K, N]
result: [batch, M, N]
```

The neural network code uses:

```text
weights: [1, outputs, inputs]
input:   [1, inputs, 1]
output:  [1, outputs, 1]
```

## Reductions

- `double tensor_sum(tensor *a)`: sum of finite values.
- `double tensor_get_max(tensor *a)`: maximum finite value.
- `double tensor_get_min(tensor *a)`: minimum finite value.
- `double tensor_get_mean(tensor *a)`: average value.
- `double tensor_argmax(tensor *a)`: index of maximum value as a `double`.
- `double tensor_argmin(tensor *a)`: index of minimum value as a `double`.
- `int tensor_argmax_index(tensor *t)`: index of maximum value as an `int`.

For classification code, prefer `tensor_argmax_index()`.

## Broadcasting

### `int broadcast_compatible(tensor *a, tensor *b)`

Checks whether dimensions are compatible for simple broadcasting.

### `tensor* tensor_apply_broadcast(tensor *a, tensor *b, double (*op)(double, double))`

Applies a binary operation with broadcasting rules. Helper operations include:

- `double add(double x, double y)`
- `double sub(double x, double y)`
- `double mul(double x, double y)`
- `double div_2(double x, double y)`

## Activations

### `void reLu(tensor *x)`

Applies ReLU in place:

```text
x = max(0, x)
```

### `void sigmoid(tensor *x)`

Applies sigmoid in place.

### `void softmax(tensor *x)`

Converts logits into probabilities in place. This is used on the final network layer for digit classification.

## Loss Helpers

### `double mean_squared_error(...)`

Computes mean squared error between two arrays.

### `double entropy(double pred[], int size)`

Computes entropy-like value from prediction probabilities.

## Printing and Loading

- `void print_tensor_rec(tensor *a, int dim, int offset)`: recursive multi-dimensional print helper.
- `void print_tensor_values(tensor *a)`: prints flat tensor values.
- `tensor load_tensor(FILE *f)`: reads one tensor from a binary model file.
