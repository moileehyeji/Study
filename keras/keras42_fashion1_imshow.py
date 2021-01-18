from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape)    #(60000, 28, 28)
print(x_test.shape)     #(10000, 28, 28)
print(y_train.shape)    #(60000,)
print(y_test.shape)     #(10000,)

print(x_train[0])
print(y_train[0])       #9

plt.imshow(x_train[0])
plt.show()              #신발



