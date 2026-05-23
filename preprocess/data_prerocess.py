import numpy as np
from torchvision import datasets

mnist = datasets.MNIST(
    root="C:/Users/Lenovo/Documents/GitHub/ML-framework-in-c/preprocess",
    train=True,
    download=True
)

X = mnist.data.numpy().astype(np.float32) / 255.0
y = mnist.targets.numpy()

X = X.reshape(-1, 784)

np.save("mnist_x.npy", X)
np.save("mnist_y.npy", y)