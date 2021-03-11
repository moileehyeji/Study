# Multi Variable
# 컬럼 두개이상

import tensorflow as tf
tf.set_random_seed(66)  # for reproducibility

x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]  #(5,3)
y_data = [[152.], [185.], [180.], [196.], [142.]]   #(5,1)

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([3,1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# hypothesis = x * w + b
hypothesis = tf.matmul(x, w) + b

#[실습]
# verbose : step, cost, hypothesis
cost = tf.reduce_mean(tf.square(hypothesis - y))

# learning_rate e-05보다 낮으면 nan
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                   feed_dict={x: x_data, y: y_data})
    if step % 10 == 0:
        print(step, "cost: ", cost_val, "\nhypothesis:\n", hy_val)

sess.close()


 