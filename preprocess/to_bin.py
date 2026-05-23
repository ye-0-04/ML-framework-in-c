import numpy as np

X = np.load("mnist_x.npy").astype(np.float32)
y = np.load("mnist_y.npy").astype(np.uint8)

X.tofile("mnist_x.bin")
y.tofile("mnist_y.bin")