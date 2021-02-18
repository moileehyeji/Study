import numpy as np

#1. DATA Load
x = np.load('./dacon/computer2/data/npy/vision_50_x3.npy')
y = np.load('./dacon/computer2/data/npy/vision_50_y3.npy')
x_pred = np.load('./dacon/computer2/data/npy//vision_50_x_pred3.npy')
print("<==complete load==>")

print(x.shape, y.shape, x_pred.shape) # (50000, 50, 50, 1) (50000, 26) (5000, 50, 50, 1)