import tensorflow as tf 
import numpy as np
tf.set_random_seed(66)

#---------------------------------------------numpy
# 텐서플로2처럼 넘파이 사용이 가능!
# csv 로드는 np.loadtxt('.csv', delimiter=',', dtype=np.float32)

dataset = np.loadtxt('C:/data/csv/wine/data-01-test-score.csv', delimiter=',', dtype=np.float32)

# [실습]
# 아래값 predict할 것
# 73,80,75,152
# 93,88,93,185
# 89,91,90,180
# 96,98,100,196
# 73,66,70,142
print(dataset.shape)    #(25, 4)

x_data = dataset[5:,:-1]
y_data = dataset[5:,-1:]
# y_data = dataset[:, [-1]]
x_test = dataset[:5,:-1]
y_test = dataset[:5,-1:]
print(x_data.shape, y_data.shape) #(20, 3) (20, 1)
print(x_test.shape, y_test.shape) #(5, 3) (5, 1)

# input
x = tf.placeholder(tf.float32, shape = [None, 3])
y = tf.placeholder(tf.float32, shape = [None, 1])

# 변수
w = tf.Variable(tf.random_normal([3,1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# 연산그래프
hypothesis = tf.matmul(x, w) + b

#loss
cost = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

# session생성
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())


#fit
for step in range(2001): 
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                   feed_dict={x: x_data, y: y_data})

    pred = sess.run(hypothesis, feed_dict={x: x_test})

    if step % 20 == 0:
        print(step, "cost: ", cost_val)  #, "\nhypothesis:\n", hy_val)

print('예측값 : \n', pred)
print('실제값 : \n', y_test)


# 예측값 :
#  [[153.1774 ]
#  [184.40659]
#  [181.38535]
#  [199.0924 ]
#  [139.50435]]
# 실제값 :
#  [[152.]
#  [185.]
#  [180.]
#  [196.]
#  [142.]]

