# Getting Started

This guide shows how to build the project, generate MNIST data, run a quick learning test, train a model, and evaluate it.

## 1. Install Requirements

You need:

- GCC
- Python
- NumPy
- torchvision

Check GCC:

```powershell
gcc --version
```

Install Python packages if needed:

```powershell
pip install numpy torchvision
```

## 2. Generate MNIST Data

Run the preprocessing scripts from the project root:

```powershell
python preprocess\data_prerocess.py
python preprocess\to_bin.py
python preprocess\fetch_test.py
```

These scripts create the local dataset files used by the C program.

Training files:

```text
mnist_x.bin
mnist_y.bin
```

Test files:

```text
mnist_test_x_v3.bin
mnist_test_y_v3.bin
```

The image files are already normalized to `0.0..1.0`, so do not divide by 255 again in C.

## 3. Compile the Framework

Compile the main program:

```powershell
gcc -O2 main.c src/tensors.c src/neural.c src/models.c -o tes.exe -Iinclude
```

Run it:

```powershell
.\tes.exe
```

## 4. Run the Fast Learning Test

Before training the full dataset, run the one-sample overfit test:

```powershell
gcc -O2 debug_one_sample.c src/tensors.c src/neural.c src/models.c -o debug_one_sample.exe -Iinclude
.\debug_one_sample.exe
```

Expected result:

- The prediction should become equal to the actual label.
- The confidence for the correct label should climb toward `1.0`.

This proves the basic forward pass, softmax, and weight update are working.

## 5. Train a Model

In `main.c`, enable the training section and choose a model shape.

Simple classifier:

```c
int shape[] = {10};
neural_network net = create_net(shape, 784, 1);
```

Hidden-layer network:

```c
int shape[] = {128, 64, 10};
neural_network net = create_net(shape, 784, 3);
```

Use `backprop_v2()` during training:

```c
forward_pass(&net, &input_tensor);
backprop_v2(&net, &target_tensor, learning_rate);
```

Save the model after training:

```c
save_model(&net, "v7_net.bin");
```

## 6. Evaluate a Model

In `main.c`, enable the evaluation section and load the saved model:

```c
neural_network net;
load_model(&net, "v7_net.bin");
```

Use the corrected v3 test files:

```c
FILE *fx = fopen("mnist_test_x_v3.bin", "rb");
FILE *fy = fopen("mnist_test_y_v3.bin", "rb");
```

Predict with:

```c
forward_pass(&net, &input_tensor);
int pred = tensor_argmax_index(&net.layers[net.size - 1].output);
```

Accuracy is:

```c
double accuracy = ((double)correct / 10000.0) * 100.0;
```

## 7. Use the Evaluation Debugger

Compile and run:

```powershell
gcc -O2 debug_eval.c src/tensors.c src/neural.c src/models.c -o debug_eval.exe -Iinclude
.\debug_eval.exe
```

This prints the first predictions, actual labels, class counts, and final accuracy. Use it when accuracy looks suspicious.

## 8. Files Not Committed

Generated dataset files are ignored by Git because they are large:

```text
mnist_*.bin
*.npy
preprocess/MNIST/
preprocess/*.idx*-ubyte
```

Anyone cloning the project should regenerate data with the preprocessing scripts instead of pushing those files.
