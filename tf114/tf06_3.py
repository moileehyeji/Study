# [실습] tf06_2_predict의 lr을 수정해서
# epoch가 2000번보다 적게 만들어
# 1000이하로 해
# W = 1.999, b = 0.999

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
train = tf.train.GradientDescentOptimizer(learning_rate = 1.7351941e-01).minimize(cost) 
# 11e-2 -> 100 5.91618e-05 [1.991303] [1.0197705]
# 112e-3-> 100 5.3595286e-05 [1.991726] [1.0188085]
# 152e-3-> 100 7.353455e-06 [1.9969654] [1.0068984]
# 1589e-4->100 5.209598e-06 [1.99745] [1.0057963]
# 1701e-4->100 2.9751945e-06 [1.9980909] [1.0043726]
# 1.7101941e-01->90 6.6796583e-06 [1.9972563] [1.0065416]
# 1.7301941e-01->90 1.4438886e-05 [1.9983088] [1.0066592]
# 1.7351941e-01->90 3.076922e-05 [1.9990958] [1.0069214]


# -----------------------------------------------------feed_dict
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(91):
        # sess.run(train)
        cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], #4개의 반환값(train은 None반환)
                                    feed_dict={x_train:[1,2,3], y_train:[3,5,7]})
        
        # -----------------------------------------------predict
        # weight와 bias가 고정된 시점에서 예측값 도출
        pred = sess.run(hypothesis, feed_dict = {x_train:[4,5,6,7,8]}) 

        if step % 10 == 0:
            #verbose
            #print(step,sess.run(cost), sess.run(W), sess.run(b))
            print(step, cost_val, W_val, b_val) 
    print('예측값 :', pred)