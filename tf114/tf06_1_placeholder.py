# input : placeholder
# sess.run :연산그래프 실행

import tensorflow as tf 

tf.set_random_seed(66)  # 랜덤값 고정


#------------------------------------------------------input
# x_train = [1,2,3]
# y_train = [3,5,7] 
x_train = tf.placeholder(tf.float32, shape = [None])
y_train = tf.placeholder(tf.float32, shape = [None])


# 텐서플로가 자체적으로 변경시키는 trainable 값
# [1] : 1차원
W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')


#-----------------------------------------------------예측
hypothesis = x_train * W + b


#------------------------------------------------------MSE
cost = tf.reduce_mean(tf.square(hypothesis - y_train))


#------------------------------------------------------경사하강법
train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost) 


# -----------------------------------------------------feed_dict
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        # sess.run(train)
        cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], #4개의 반환값(train은 None반환)
                                    feed_dict={x_train:[1,2,3], y_train:[3,5,7]})

        if step % 20 == 0:
            #verbose
            # print(step,sess.run(cost), sess.run(W), sess.run(b))
            print(step, cost_val, W_val, b_val)   
 