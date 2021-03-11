import tensorflow as tf

x = [1,2,3]
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([1.0], tf.float32)

hypothesis = W * x + b

# [실습]
# 1. sess.run()
# 2. InteractiveSession
# 3. .eval(session=sess)
# print('hypothesis: ', ???)


#---------------------------------1
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(hypothesis)
print('hypothesis1 : ', aaa) #hypothesis :  [1.3       1.6       1.9000001]
sess.close()


#---------------------------------2
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = hypothesis.eval()
print('hypothesis2 : ',bbb) #hypothesis2 :  [1.3       1.6       1.9000001]
sess.close()


#---------------------------------3
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = hypothesis.eval(session = sess)
print('hypothesis3 : ',ccc) #hypothesis3 :  [1.3       1.6       1.9000001]
sess.close()