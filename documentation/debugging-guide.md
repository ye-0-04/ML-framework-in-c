# Debugging Guide

This guide gives fast checks for training problems without waiting for full MNIST epochs.

## Why Debug Small First

Full MNIST training is slow while the framework is still changing. A tiny test can show whether the math works in seconds.

Use this order:

1. One sample with a `784 -> 10` network.
2. One sample with hidden layers.
3. A small subset, such as 100 samples.
4. Full MNIST training.

## One-Sample Overfit Test

Use:

```text
debug_one_sample.c
```

Compile:

```powershell
gcc -O2 debug_one_sample.c src/tensors.c src/neural.c src/models.c -o debug_one_sample.exe -Iinclude
```

Run:

```powershell
.\debug_one_sample.exe
```

Expected behavior:

```text
epoch=0 pred=... actual=... confidence=...
epoch=100 pred=<actual> actual=<actual> confidence=high
...
epoch=999 pred=<actual> actual=<actual> confidence close to 1.0
```

If this fails, the issue is in core learning:

- forward pass
- output gradient
- weight update
- target one-hot encoding
- input loading

If this passes, the basic single-layer training path works.

## Evaluation Debugger

Use:

```text
debug_eval.c
```

Compile:

```powershell
gcc -O2 debug_eval.c src/tensors.c src/neural.c src/models.c -o debug_eval.exe -Iinclude
```

Run:

```powershell
.\debug_eval.exe
```

It prints:

- first 20 predictions
- first 20 actual labels
- probability of the correct class
- max predicted probability
- total correct count
- actual class counts
- predicted class counts

## Interpreting Accuracy

For MNIST:

```text
~10% accuracy: random guessing
0% to 1% accuracy: usually a data/evaluation bug
>80% after a simple training setup: learning is working
>90%: good sign for this project
```

Be careful with percent formatting:

```text
0.96% is bad
9.60% is random-ish
96.00% is excellent
```

## Common Problems

### Using Broken Test Labels

Do not evaluate with:

```text
mnist_test_y_v2.bin
```

Use:

```text
mnist_test_y_v3.bin
```

The v2 label file has the wrong size and wrong data representation.

### Evaluating the Wrong Network Variable

If the model is loaded into `net_2`, evaluation must use `net_2`:

```c
forward_pass(&net_2, &input_tensor);
int pred = tensor_argmax_index(&net_2.layers[net_2.size - 1].output);
```

Do not accidentally run `forward_pass()` on an older in-memory `net`.

### Learning Rate Too Small

`0.0001` can make learning look frozen.

Try:

```c
double learning_rate = 0.01;
```

### Learning Rate Too Large

`0.1` can overshoot or collapse predictions.

Symptoms:

- accuracy falls to zero
- predicted counts show almost every sample predicted as one class
- probabilities become extreme

### Printing Too Much

These slow training heavily:

```c
printf("\033[2J\033[H");
printf("%i", i);
printf("Epoch %i done\n", epoch); /* inside the sample loop */
```

Prefer printing every 1000 samples or once per epoch.

### Double Normalization

The current `.bin` image files are already normalized to `0.0..1.0`.

Do not divide by 255 again in C when reading `mnist_x.bin` or `mnist_test_x_v3.bin`.

## Good Debug Print

During evaluation, print the first few rows:

```c
if (i < 20)
{
    printf("i=%d pred=%d actual=%d\n", i, pred, actual);
}
```

The first 20 actual labels should be:

```text
7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4
```

If not, check the label file path and type.
