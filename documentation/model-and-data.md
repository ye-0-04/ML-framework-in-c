# Model and Data Files

This document explains the binary model format and the MNIST preprocessing flow used by the project.

## Model Serialization

Model save/load is implemented in `src/models.c`.

### `void save_model(neural_network *nt, const char *filename)`

Writes a network to a binary file.

File layout:

```text
network size: int

for each layer:
    weights.dims: int
    weights.size: int
    weights.shape: int[weights.dims]
    weights.stride: int[weights.dims]
    weights.data: double[weights.size]

    bias.dims: int
    bias.size: int
    bias.shape: int[bias.dims]
    bias.stride: int[bias.dims]
    bias.data: double[bias.size]
```

Only trainable parameters are saved:

- weights
- biases

Temporary training tensors are not saved:

- input
- output
- gradient

### `void load_model(neural_network *nt, const char *filename)`

Reads a saved network from disk.

For each loaded layer:

- weights are loaded from the file.
- biases are loaded from the file.
- input, output, and gradient are initialized as empty tensors.

After loading, call `forward_pass()` before reading layer outputs.

## MNIST Training Files

Training data is produced by:

```text
preprocess/data_prerocess.py
preprocess/to_bin.py
```

The training image file is:

```text
mnist_x.bin
```

Expected format:

```text
60000 * 784 float32 values
```

Images are already normalized in Python:

```python
X = mnist.data.numpy().astype(np.float32) / 255.0
```

The training label file is:

```text
mnist_y.bin
```

Expected format:

```text
60000 uint8 values
```

Each label should be an integer from `0` to `9`.

## MNIST Test Files

Test data is produced by:

```text
preprocess/fetch_test.py
```

The corrected test files are:

```text
mnist_test_x_v3.bin
mnist_test_y_v3.bin
```

Expected formats:

```text
mnist_test_x_v3.bin: 10000 * 784 float32 values
mnist_test_y_v3.bin: 10000 uint8 values
```

The test images are already normalized in Python:

```python
X_test_v3 = mnist_test.data.numpy().astype(np.float32) / 255.0
```

The test labels must not be divided by 255. They should stay as `uint8` class IDs:

```python
y_test_v3 = mnist_test.targets.numpy().astype(np.uint8)
```

## Known Bad File

`mnist_test_y_v2.bin` is not a valid uint8 label file.

It was produced from labels that were divided by 255 and written with the wrong numeric representation. Its size is `80000` bytes instead of `10000` bytes.

Do not use it for evaluation.

Use:

```c
FILE *fx = fopen("mnist_test_x_v3.bin", "rb");
FILE *fy = fopen("mnist_test_y_v3.bin", "rb");
```

## Label Sanity Check

The first 20 MNIST test labels should be:

```text
7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4
```

If evaluation prints labels outside `0..9`, the wrong label file is being read.

## Input Scaling Rule

When using the current `.bin` files:

```c
input_tensor.data[j] = X_test[i * 784 + j];
```

Do not do:

```c
input_tensor.data[j] = X_test[i * 784 + j] / 255.0;
```

The division already happened in preprocessing.
