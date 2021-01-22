# dacon quantile_loss 함수
'''
def quantile_loss (q, y, pred):
    err = (y-pred)
    return mean(maximum(q*err, (q-1)*err), axis=-1)
'''

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

# mse 함수정의 :  compile loss에서 사용시 인자전달 안해줘도 자동으로 넣어줌
def custom_mse(y_true, y_pred):                 
    return tf.math.reduce_min(tf.square(y_true-y_pred))     #평균(제곱(차))


#1.데이터
x = np.array([1,2,3,4,5,6,7,8]).astype('float32')
y = np.array([1,2,3,4,5,6,7,8]).astype('float32')
print(x.shape)

# astype('float32') --->  TypeError: Input 'y' of 'Sub' Op has type float32 that does not match type int32 of argument 'x'.

#2.모델
model = Sequential()
model.add(Dense(10, input_shape=(1,)))
model.add(Dense(10))
model.add(Dense(1))

#3.컴파일
model.compile(loss=custom_mse, optimizer='adam', metrics='mae') #custom_mse 인자는 훈련시 자동으로 넣어줌
model.fit(x, y, epochs=8, batch_size=1)

#4.평가, 예측
loss = model.evaluate(x, y)

print(loss)

'''
custom_mse : 
[0.000549785909242928, 1.483818769454956]
'''