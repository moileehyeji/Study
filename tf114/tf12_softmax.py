# softmax
import tensorflow as tf 
import numpy as np
tf.set_random_seed(66)

#input
x_data = [[1,2,1,1],[2,1,3,2],[3,1,3,4],[4,1,5,5],[1,7,5,5],[1,2,5,6],[1,6,6,6],[1,7,6,7]]
y_data = [[0,0,1],  [0,0,1],  [0,0,1],  [0,1,0],  [0,1,0],  [0,1,0],  [1,0,0],  [1,0,0]]


x = tf.placeholder('float', [None,4])   #None: 같은규격의 행은 더 늘릴 수 있다.
y = tf.placeholder('float', [None,3])   #sigmoid : [None,1]


#---------------------------------------------------------
# w: category 3이므로 [4,3]
# b: 행렬의 덧셈은 쉐이프가 같아야 함
# bias는 1layer에 통상 1개 
# weight 하나에 bias 스칼라 1개
w = tf.Variable(tf.random_normal([4,3]), name = 'weight')   
b = tf.Variable(tf.random_normal([1,3]), name = 'bias')

#---------------------------------------------------------Activation(hypothesis)
# hypothesis의 shape(8,3)
# [None,4] * [4,3] + [1,3] = [None,3] + [1,3] = [None,3]
# 레이어는 activation으로 감싸서 output
hypothesis = tf.nn.softmax(tf.matmul(x,w) + b)


#----------------------------------------------------------loss
# cost = tf.reduce_mean(tf.square(hypothesis - y))    #mse
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1)) #categorical_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(loss)

#fit
with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        _, cost_val = sess.run([optimizer,loss], feed_dict = {x:x_data, y:y_data})

        if step % 200 == 0:
            print(step, 'cost_val : ', cost_val)

    #predict
    a = sess.run(hypothesis, feed_dict = {x:[[1, 11, 7, 9]]})
    print(a, sess.run(tf.argmax(a,1))) #tf.arg_max : 가장 높은 값에 1부여  
    # [[0.80384046 0.19088006 0.00527951]] [0] 


#with문 밖에서 sess.run사용하면 RuntimeError: Attempted to use a closed Session.