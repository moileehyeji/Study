# 텐서머신 만들자!
# y = wx + b
# x : placeholder
# x, b: variable
# 성킴교수 강의 참고하심

import tensorflow as tf 

tf.set_random_seed(66)  # 랜덤값 고정

x_train = [1,2,3]
y_train = [3,5,7] 


#random_normal: 정규 분포로 텐서를 생성하는 이니셜 라이저.
# 텐서플로가 자체적으로 변경시키는 trainable 값
# [1] : 1차원
W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(W), sess.run(b)) #[0.06524777] [1.4264158]


#-----------------------------------------------------예측값
#hyperthesis : 예측, 가설
hyperthesis = x_train * W + b
#             리스트    variable


#------------------------------------------------------MSE
#loss         평균                예측값       실제값
cost = tf.reduce_mean(tf.square(hyperthesis - y_train))

#optimizer - 경사하강법
#: y축 cost, x축 weight인 그래프의 한 지점의 기울기가
#양수일때는 -weight, 음수일때는 weight
# weight 조정하는 수식은 하단
# 이것을 해주는 함수가 GradientDescentOptimizer, minimize
""" 
# Minimize: Gradient Descent using derivative: W -= learning_rate * derivative
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent) 
"""
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)

#loss가 minimize될 때까지 훈련시키겠다!
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        #verbose
        print(step,sess.run(cost), sess.run(W), sess.run(b))
        #     epoch,loss,           가중치,      절편   
 