import numpy as np
from torchvision import datasets

mnist_test = datasets.MNIST(
    root="C:/Users/Lenovo/Documents/GitHub/ML-framework-in-c/preprocess",
    train=False,
    download=True
)

X_test_v3 = mnist_test.data.numpy().astype(np.float32) / 255.0
y_test_v3 = mnist_test.targets.numpy().astype(np.uint8)
X_test_v3 = X_test_v3.reshape(-1, 784)

X_test_v3.tofile("mnist_test_x_v3.bin")
y_test_v3.tofile("mnist_test_y_v3.bin")