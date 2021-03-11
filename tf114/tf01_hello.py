import tensorflow as tf
print(tf.__version__)  

# 텐서플로 3가지 자료형
# 1. constant
# 2. placeholder
# 3. variable

hello = tf.constant('Hello World')  #3가지 자료형중 하나 node를 만들어주는 개념
print(hello)                        #해당 자료형의 구조만 출력Tensor("Const:0", shape=(), dtype=string)
                                    #WHY?

sess = tf.Session()     # 상단의 값을 출력하려면 Session을 거쳐야함
print(sess.run(hello))  #b'Hello World'