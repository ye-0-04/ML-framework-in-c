# Neural Network API Documentation

The neural module builds dense feed-forward neural networks on top of the tensor module.

## Data Structures

### `layer`

```c
typedef struct layer {
    tensor weights;
    tensor bias;
    tensor output;
    tensor input;
    tensor gradient;
} layer;
```

- `weights`: dense weight matrix shaped `[1, outputs, inputs]`.
- `bias`: bias vector shaped `[1, outputs, 1]`.
- `input`: cached input for the layer from the latest forward pass.
- `output`: cached output for the layer from the latest forward pass.
- `gradient`: cached gradient used during backpropagation.

### `neural_network`

```c
typedef struct neural_network {
    layer *layers;
    int size;
} neural_network;
```

- `layers`: heap array of layers.
- `size`: number of layers.

## Layer Creation

### `layer create_layer(int inputs, int outputs)`

Creates one dense layer.

Internally:

- `weights` shape is `[1, outputs, inputs]`.
- `bias` shape is `[1, outputs, 1]`.
- weights and biases are initialized with `rand_fill()`.
- `gradient`, `input`, and `output` start as empty tensors.

Example:

```c
layer l = create_layer(784, 10);
```

This creates a layer that maps 784 input values to 10 output values.

## Network Creation

### `neural_network create_net(int neural_shape[], int input, int size)`

Creates a stack of dense layers.

Example:

```c
int shape[] = {128, 64, 10};
neural_network net = create_net(shape, 784, 3);
```

This builds:

```text
784 -> 128 -> 64 -> 10
```

For a single-layer classifier:

```c
int shape[] = {10};
neural_network net = create_net(shape, 784, 1);
```

This builds:

```text
784 -> 10
```

## Forward Pass

### `void forward_pass(neural_network *nt, tensor *input)`

Runs inference through all layers.

Flow:

1. Clone the input tensor into the first layer input cache.
2. For each layer, compute `weights * input`.
3. Add the layer bias.
4. Apply ReLU for hidden layers.
5. Apply softmax for the final layer.
6. Cache each layer output for backpropagation.

Expected input shape:

```text
[1, input_size, 1]
```

For MNIST:

```text
[1, 784, 1]
```

Final output for digit classification:

```text
[1, 10, 1]
```

Use `tensor_argmax_index(&net.layers[net.size - 1].output)` to get the predicted class.

## Backpropagation

### `void backprop_v2(neural_network *nt, tensor *correct_output, double l_r)`

Trains the network using the cached values from the latest `forward_pass()`.

Expected target shape:

```text
[1, 10, 1]
```

The target should be one-hot encoded:

```text
label 5 -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
```

Output-layer gradient:

```text
prediction - target
```

Hidden-layer gradient:

```text
transpose(weights_next) * gradient_next
```

Hidden layers also apply the ReLU derivative. The final softmax layer should not use the ReLU derivative.

Parameter update:

```text
weights = weights - learning_rate * gradient * transpose(input)
bias    = bias    - learning_rate * gradient
```

Use `backprop_v2()` for current training. `backprop()` is older and does not contain the same fixes.

## Training Loop Pattern

```c
for (int epoch = 0; epoch < EPOCHS; epoch++)
{
    for (int i = 0; i < TRAIN_SIZE; i++)
    {
        load_input_sample(&input_tensor, X_train, i);
        load_label(&target_tensor, y_train[i]);

        forward_pass(&net, &input_tensor);
        backprop_v2(&net, &target_tensor, learning_rate);
    }
}
```

For debugging, start with:

```c
int shape[] = {10};
double learning_rate = 0.01;
```

Then move to hidden layers after the single-layer network learns.

## Evaluation Pattern

```c
forward_pass(&net, &input_tensor);
int pred = tensor_argmax_index(&net.layers[net.size - 1].output);
```

Compare `pred` with the integer label from the test label file.

## Practical Notes

- The input `.bin` files produced by the preprocessing scripts are already normalized to `0.0..1.0`.
- Do not divide by 255 again in C when using those files.
- Learning rate matters a lot. `0.0001` can be too slow, while `0.1` can be unstable.
- Printing every sample can make training extremely slow. Print every 1000 samples or per epoch instead.
