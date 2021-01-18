from tensorflow.keras.datasets import cifar100
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape)    #(50000, 32, 32, 3)
print(x_test.shape)     #(10000, 32, 32, 3)
print(y_train.shape)    #(50000, 1)
print(y_test.shape)     #(10000, 1)

print(x_test[0])
print(y_test)

plt.imshow(x_test[0])
plt.show()
