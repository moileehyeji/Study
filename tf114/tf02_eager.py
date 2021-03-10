
import tensorflow as tf

#-------------------------------------------------즉시 실행 모드
''' #tensorflow의 Tensorflow 2.에서 1.대 사용가능하도록
# from tensorflow.python.framework.ops import disable_eager_execution
print(tf.executing_eagerly())   #True : Tensorflow 2.대니까 트루(1.대면 false)
tf.compat.v1.disable_eager_execution()  #즉시실행모드 on
print(tf.executing_eagerly())   #False  '''
#-------------------------------------------------

print(tf.__version__) 
 

# 텐서플로 3가지 자료형
# 1. constant
# 2. 입력값?
# 3. variable

hello = tf.constant('Hello World')  #3가지 자료형중 하나
print(hello)                        #해당 자료형의 구조만 출력Tensor("Const:0", shape=(), dtype=string)
                                    #WHY?

# sess = tf.Session()           # tensorflow 1.13버전
sess = tf.compat.v1.Session()   # tensorflow 1.14버전
print(sess.run(hello))  #b'Hello World'


#즉시실행모드 하기 전 AttributeError: module 'tensorflow' has no attribute 'Session'
#즉시실행모드 한 후