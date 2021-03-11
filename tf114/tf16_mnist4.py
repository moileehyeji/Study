# 가중치 초기화

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

x_train = x_train.reshape(-1, 784)/255.
x_test = x_test.reshape(-1, 784)/255.
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

#전처리
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



x = tf.placeholder(tf.float32, shape = [None, 784])
y = tf.placeholder(tf.float32, shape = [None, 10])

# 2. 모델구성
n1 = 100
n2 = 50
n3 = 50


#---------------------------------------------------------------------------------------가중치 초기화
# tf.get_variable : 전달된 이니셜 라이저 사용
# w1 = tf.Variable(tf.random_normal([784,n1], stddev=0.1), name = 'weight1')
#relu
w1 = tf.get_variable('weight1', shape = [784,n1], initializer = tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([n1], stddev=0.1), name = 'bias1')
layer1 = tf.nn.selu(tf.matmul(x, w1) + b1)
layer1 = tf.nn.dropout(layer1, keep_prob=0.3)

# layer1 info 출력--------------------------------------------------
print('w1 : ', w1)  
print('b1 : ', b1) 
print('layer1 : ', layer1) 
# w1 :  <tf.Variable 'w1:0' shape=(784, 100) dtype=float32_ref> 
# b1 :  <tf.Variable 'bias1:0' shape=(100,) dtype=float32_ref>
# layer1 :  Tensor("Elu:0", shape=(?, 100), dtype=float32)
# layer1 = tf.nn.dropout(layer1, keep_prob=0.3)
# ------------------------------------------------------------------
#hidden
#elu
w2 = tf.get_variable('weight2', shape = [n1,n2], initializer = tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([n2], stddev=0.1), name = 'bias2')
layer2 = tf.nn.selu(tf.matmul(layer1, w2) + b2)
layer2 = tf.nn.dropout(layer2, keep_prob=0.3)
#selu
w3 = tf.get_variable('weight3', shape = [n2,n3], initializer = tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([n3], stddev=0.1), name = 'bias3')
layer3 = tf.nn.selu(tf.matmul(layer2, w3) + b3)
layer3 = tf.nn.dropout(layer3, keep_prob=0.3)
# output
w4 = tf.get_variable('weight4', shape = [n3,10], initializer = tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([10], stddev=0.1), name = 'bias4')
hypothesis = tf.nn.softmax(tf.matmul(layer3, w4) + b4)


# 3. 컴파일, 훈련
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))
# train = tf.train.GradientDescentOptimizer(learning_rate = 1e-3).minimize(cost)
train = tf.train.AdamOptimizer(learning_rate = 1.5e-5).minimize(cost)

#-----------------------------------------------------------------batch_size
# 문제 : 훈련을 데이터 통으로 하고 있어
# batch_size 추가하자
# 메모리터짐 방지
training_epochs = 150
batch_size = 100
total_batch = int(len(x_train)/batch_size)  #60000 / 100 = 600  -> 1epoch에 600번

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    avg_cost = 0

    for i in range(total_batch):
        start = i * batch_size  #0
        end = start + batch_size#100

        batch_x, batch_y = x_train[start:end], y_train[start:end]
        feed_dict = {x:batch_x, y:batch_y}
        c, _ = sess.run([cost, train], feed_dict = feed_dict)   #100개(batch_size) 훈련한 cost
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
print('acc : ', sess.run(accuracy, feed_dict={x:x_test, y:y_test})) # acc :  0.7586


# Epoch :     6 cost  :1.136741024
# Epoch :     7 cost  :1.040545466
# Epoch :     8 cost  :0.959405109
# Epoch :     9 cost  :nan
# Epoch :    10 cost  :nan
# Epoch :    11 cost  :nan
# Epoch :    12 cost  :nan
# 기울기 소실을 완화하는 가장 간단한 방법은 은닉층의 활성화 함수로 시그모이드나 하이퍼볼릭탄젠트 함수 대신에 ReLU나 ReLU의 변형 함수와 같은 Leaky ReLU를 사용하는 것입니다.
# 기울기가 거의 0으로 소멸되어 버리면 네트워크의 학습은 매우 느려지고, 학습이 다 이루어지지 않은 상태에서 멈출 것입니다. 이를 지역 최솟값에 도달한다고 표현하기도 합니다.
# 이를 해결하기 위한 방법으로 사라져가는 성질을 갖지 않는 비선형 함수를 활성화 함수(예 : ReLu함수)로 선택하면 해결할 수 있습니다.

