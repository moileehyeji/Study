import tensorflow as tf 

sess = tf.Session()


#---------------------------------------------------variable
# global_variable_initailizer 초기화가 필수 -> 아무 의미 없어 tensorflow에서 쓸수 있게 전역 변수를 초기화해주는거


x = tf.Variable([2], dtype = tf.float32, name='test')
print(x) #<tf.Variable 'test:0' shape=(1,) dtype=float32_ref>   
#WARNING:The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.

# 전역 변수를 초기화
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(x))