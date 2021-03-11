# [실습]
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf


(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

x_train = x_train.reshape(-1, 784)/255.
x_test = x_test.reshape(-1, 784)/255.
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

#전처리
y_train = to_categorical(y_train)

# input
x = tf.placeholder(tf.float32, shape = [None, 784])
y = tf.placeholder(tf.float32, shape = [None, 10])

#hidden
n1 = 64
n2 = 128

w1 = tf.Variable(tf.random_normal([784,n1], stddev=0.1), name = 'weight1')
b1 = tf.Variable(tf.random_normal([n1], stddev=0.1), name = 'bias1')
layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = tf.Variable(tf.random_normal([n1,n2], stddev=0.1), name = 'weight2')
b2 = tf.Variable(tf.random_normal([n2], stddev=0.1), name = 'bias2')
layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)

# output
w4 = tf.Variable(tf.random_normal([n1,10], stddev=0.1), name = 'weight4')
b4 = tf.Variable(tf.random_normal([10], stddev=0.1), name = 'bias4')
hypothesis = tf.nn.softmax(tf.matmul(layer1, w4) + b4)


cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))

train = tf.train.GradientDescentOptimizer(learning_rate = 1e-1).minimize(cost)
# train = tf.train.AdamOptimizer(learning_rate = 1e-6).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(151):
        cost_val, _ = sess.run([cost, train], feed_dict = {x:x_train, y:y_train})

        if step % 1 == 0:
            print(step, cost_val)

    # predict
    pred = sess.run(hypothesis, feed_dict = {x:x_test})
    pred = sess.run(tf.argmax(pred, 1))
    print('pred  : ', pred[:7])
    print('y_test:', y_test[:7])

    #accuracy_score
    acc = accuracy_score(pred, y_test)
    print('acc : ', acc)    #acc :  0.8796