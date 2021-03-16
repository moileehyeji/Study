# cpu 너무 느리다!
# base환경으로 변경한 뒤
# 즉시실행모드 해보자
# --> AttributeError: module 'tensorflow' has no attribute 'placeholder' 
# --> tf.compat.v1.placeholder   
# --> 이런 에러가 계속 뜨면 .compat.v1추가하거나 주석
import tensorflow as tf
import numpy as np 
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler

#-------------------------------------------------즉시 실행 모드
#tensorflow의 Tensorflow 2.에서 1.대 사용가능하도록
# from tensorflow.python.framework.ops import disable_eager_execution
tf.compat.v1.disable_eager_execution()  #즉시실행모드 on
print(tf.executing_eagerly())   #False  

print(tf.__version__) #2.4.1
#-------------------------------------------------

# 추가적인 튜닝 해보자  (acc 0.98이상)
# nan이 문제
# tf.set_random_seed(66)

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)#(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)  #(10000, 32, 32, 3) (10000, 1)

#전처리
# sc = MinMaxScaler()
# x_train = x_train.reshape(-1,28*28)
# x_test = x_test.reshape(-1,28*28)
# sc.fit(x_train)
# x_train = sc.transform(x_train)
# x_test = sc.transform(x_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(-1,32,32,3)/255.
x_test = x_test.reshape(-1,32,32,3)/255.

x = tf.compat.v1.placeholder(tf.float32, [None,32,32,3])
y = tf.compat.v1.placeholder(tf.float32, [None,10])

# 2. 모델구성
# -----------------------------------------------------------------------------------CNN    
#Conv2D(filter, kernel_size, input_shape)
#Conv2D(32, (3,3), input_shape(28,28,1))
#param# = (input_dim(chennel) * kernel_size + bias) * filters = 50

# 코드 이해
# Tensor형 weight변수
# Tensor형 연산변수
# bias는 기본값이 있을 것
# w1의 shape = [3,3,1,32]
# ->shape = [kernel_size[0],kernel_size[1],input(chennel),output(filters)]
# L1의 strides = [1,1,1,1]
# -> strides = [shape set, strides, strides, shape set]
# padding = 'SAME' --> output(28,28,32)
# padding = 'VALID' --> output(26,26,32)
n1 = 32
n2 = 64
n3 = 128
n4 = 64
n5 = 32
n6 = 16
# initializer1 = tf.contrib.layers.xavier_initializer()
# initializer2 = tf.keras.initializers.he_uniform()
#--------------Conv2D
w1 = tf.compat.v1.get_variable('w1', shape = [3,3,3,n1])
L1 = tf.nn.conv2d(x, w1, strides = [1,1,1,1], padding = 'SAME') #shape=(?, 32, 32, 32)
L1 = tf.nn.selu(L1)
L1 = tf.nn.max_pool(L1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME') #shape=(?,16,16,32)
print(L1)
w2 = tf.compat.v1.get_variable('w2', shape = [3,3,n1,n2])
L2 = tf.nn.conv2d(L1, w2, strides = [1,1,1,1], padding = 'SAME') #shape=(?,16,16,64)
L2 = tf.nn.selu(L2)
L2 = tf.nn.max_pool(L2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME') #shape=(?,8,8,64)
print(L2)
w3= tf.compat.v1.get_variable('w3', shape = [3,3,n2,n3])
L3 = tf.nn.conv2d(L2, w3, strides = [1,1,1,1], padding = 'SAME') #shape=(?,8,8,128)
L3 = tf.nn.selu(L3)
L3 = tf.nn.max_pool(L3, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME') #shape=(?,4,4,128)
print(L3)
w4= tf.compat.v1.get_variable('w4', shape = [3,3,n3,n4])
L4 = tf.nn.conv2d(L3, w4, strides = [1,1,1,1], padding = 'SAME') #shape=(?,4,4,64)
L4 = tf.nn.selu(L4)
L4 = tf.nn.max_pool(L4, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME') #shape=(?,2,2,64)
print(L4) 
#--------------Flatten
#L4의 shape=(?,2,2,64)
L_flat = tf.reshape(L4, [-1,2*2*n4]) #shape=(?, 256)
print(L_flat) 
#--------------Dense
w5 = tf.compat.v1.get_variable('w5', shape = [2*2*n4,n5])#, initializer = initializer1)
b5 = tf.Variable(tf.compat.v1.random_normal([n5], name = 'b1'))
L5 = tf.nn.selu(tf.matmul(L_flat,w5) + b5)
# L5 = tf.nn.dropout(L5, keep_prob=0.2) #shape=(?, 32)
print(L5)
w6 = tf.compat.v1.get_variable('w6', shape = [n5,n6])#, initializer = initializer1)
b6 = tf.Variable(tf.compat.v1.random_normal([n6], name = 'b2'))
L6 = tf.nn.selu(tf.matmul(L5,w6) + b6)
# L6 = tf.nn.dropout(L6, keep_prob=0.2) #shape=(?, 16)
print(L6)
#--------------Output
w7 = tf.compat.v1.get_variable('w7', shape = [n6,10])#, initializer = initializer1)
b7 = tf.Variable(tf.compat.v1.random_normal([10], name = 'b3'))
hypothesis = tf.nn.softmax(tf.matmul(L6,w7) + b7) #shape=(?, 10)
print(hypothesis)


# 3. 컴파일, 훈련
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis), axis=1))
optimizer = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

learning_rate = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train)/batch_size)

for epoch in range(training_epochs):
    avg_cost = 0

    for i in range(total_batch):
        start = i * batch_size  #0
        end = start + batch_size#100

        batch_x, batch_y = x_train[start:end], y_train[start:end]
        feed_dict = {x:batch_x, y:batch_y}
        c, _ = sess.run([loss, optimizer], feed_dict = feed_dict)   #100개(batch_size) 훈련한 cost
        avg_cost += c   # 실질적인 600개 cost출력

        # 훈련 100개(batch_size)씩 =c
        # 600번 하지 =total_batch
        # 모두 통합해서 저장해서 /600하면 cost의 평균이 나와 =avg_cost
    avg_cost /= total_batch
    print('Epoch : ', '%4d' %(epoch + 1), 
          'cost  :{:.9f}'.format(avg_cost)) 

print('훈련 끝')

prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('acc : ', sess.run(accuracy, feed_dict={x:x_test, y:y_test}))     #acc :  0.6692


# selu
# relu
# tf.keras.initializers.GlorotNormal())