import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1.데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

# 2.모델
model = Sequential()
model.add(Dense(4, input_dim = 1))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

model.summary()
# 연산수 Total params: 34
# Total params: 34
# Trainable params: 34      ---> 훈련시키는 weight
# Non-trainable params: 0   ---> 훈련안하는 weight (전이학습모델의 경우 가중치가 이미 저장된 상태라 Non-trainable params발생)    

# print(model.weights)
'''
[<tf.Variable 'dense/kernel:0' shape=(1, 4) dtype=float32, numpy=
array([[ 0.06370652,  0.01566708, -0.22690785,  0.39245796]],       
      dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(4,) dtype=float32, numpy=array([0., 0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(4, 3) dtype=float32, numpy=
      ------> 4개 (다음 weight로 전달 해줌)
      ------>'dense/bias:0' shape=(4,) dtype=float32, numpy=array([0., 0., 0., 0.]

array([[ 0.59911823, -0.41859132, -0.3545257 ],
       [ 0.42075717,  0.21247113,  0.37085545],
       [ 0.8381289 ,  0.26877236,  0.01463282],
       [ 0.7838342 , -0.7724058 , -0.10465622]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(3, 2) dtype=float32, numpy=
       ------> 3*4개
       ------>'dense_1/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.]

array([[ 0.3767792 , -0.95703155],
       [ 0.27434707,  0.2858466 ],
       [-0.6177709 ,  0.48175037]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_3/kernel:0' shape=(2, 1) dtype=float32, numpy=     
       ------> 2*3개
       ------> 'dense_2/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.]

array([[-0.18522632],
       [ 1.1535219 ]], dtype=float32)>, <tf.Variable 'dense_3/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
       ------> 1*2개
       ------>'dense_3/bias:0' shape=(1,) dtype=float32, numpy=array([0.]

'''

print(model.trainable_weights) #훈련시키는 weight값 
# model.weight와 동일 
# WHY? layer 모두 훈련시키므로
'''
[<tf.Variable 'dense/kernel:0' shape=(1, 4) dtype=float32, numpy=
array([[-0.5980522 , -0.12704939, -0.36729908, -0.38692713]],
      dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(4,) dtype=float32, numpy=array([0., 0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(4, 3) dtype=float32, numpy=

array([[ 0.07192796,  0.57081044,  0.5548245 ],
       [-0.09555566, -0.5109152 , -0.22014797],
       [ 0.36998582,  0.25465178,  0.44743955],
       [-0.8243301 , -0.23109812,  0.57789016]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(3, 2) dtype=float32, numpy=

array([[-0.9783933 ,  0.54637206],
       [ 0.38116086,  0.07822955],
       [-0.75406265,  0.6237346 ]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_3/kernel:0' shape=(2, 1) dtype=float32, numpy=     

array([[-0.2332784 ],
       [ 0.23605573]], dtype=float32)>, <tf.Variable 'dense_3/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]

'''
print(len(model.weights))   #8  (1 layer = waight1 + bias1)
print(len(model.trainable_weights)) #8
# 레이어 하나 늘릴경우 10

