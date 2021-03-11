# 인공지능의 겨울
# 다층레이어 구성
# 레이어마다 Weight와 Bias의 shape
# 연산그래프 수식의 x값 == 전 레이어의 연산그래프 
import tensorflow as tf 
import numpy as np 
tf.set_random_seed(66)

x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype = np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype = np.float32)

#--------------------------------------------------------------input layer
#output node: 2
x = tf.placeholder(tf.float32, shape = [None, 2])
y = tf.placeholder(tf.float32, shape = [None, 1])
#--------------------------------------------------------------hidden layer1
#input  : x
#output node: 10
#model.add(Dense(10, input_dim=2, activation = 'sigmoid'))
w1 = tf.Variable(tf.random_normal([2,10]), name = 'weight11')
b1 = tf.Variable(tf.random_normal([10]), name = 'bias1')
layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)
#--------------------------------------------------------------hidden layer2
#input : layer1
#output node: 7
#model.add(Dense(7, activation = 'sigmoid'))
w2 = tf.Variable(tf.random_normal([10,7]), name = 'weight2')
b2 = tf.Variable(tf.random_normal([7]), name = 'bias2')
layer2 = tf.sigmoid(tf.matmul(layer1,w2) + b2)
#--------------------------------------------------------------output layer
#input : layer2
#output node: 1
#model.add(Dense(1, activation = 'sigmoid'))
w3 = tf.Variable(tf.random_normal([7,1]), name = 'weight3')
b3 = tf.Variable(tf.random_normal([1]), name = 'bias3')
hypothesis = tf.sigmoid(tf.matmul(layer2,w3) + b3)
#--------------------------------------------------------------


cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate = 2.1e-2).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype = tf.float32))


#fit, output
with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer())

    for step in range(5001): 
        cost_val, _ = sess.run([cost, train], feed_dict = {x:x_data, y:y_data})

        if step % 200 == 0: 
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict = {x:x_data, y:y_data})

    print('예측값 : ', h, '\n 원래값 : ', c, '\n Accuracy : ', a)   # Accuracy :  1.0