import tensorflow as tf
tf.set_random_seed(777)

# Variable 사용법 3가지
W = tf.Variable(tf.compat.v1.random_normal([1]), name = 'weight')
print(W)    #<tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>


# 워닝 없애기
# WARNING:tensorflow:From c:\Study\tf114\tf07_Variable.py:9: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead
#WARNING:tensorflow:From c:\Study\tf114\tf07_Variable.py:14: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
# tf.---->tf.compat.v1. 수정

#---------------------------------1
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(W)
print('aaa : ', aaa) #aaa :  [2.2086694]
sess.close()


#---------------------------------2
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = W.eval()
print('bbb : ',bbb) #bbb :  [2.2086694]
sess.close()


#---------------------------------3
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = W.eval(session = sess)
print('ccc : ',ccc) #ccc :  [2.2086694]
sess.close()
