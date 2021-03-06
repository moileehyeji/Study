# metrics가 다를경우

#1. 데이터
import numpy as np

x1 = np.array([range(100), range(301,401), range(1,101)])
y1 = np.array([range(711,811), range(1,101), range(201,301)])

x2 = np.array([range(101,201), range(411,511), range(100,200)])
y2 = np.array([range(501,601), range(711,811), range(100)])

x1=np.transpose(x1)
x2=np.transpose(x2)
y1=np.transpose(y1)
y2=np.transpose(y2)

from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=0.8, shuffle=False)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, train_size=0.8, shuffle=False)


#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense

input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu')(input1)
dense1 = Dense(30, activation='relu')(dense1)
dense1 = Dense(50, activation='relu')(dense1)
dense1 = Dense(30, activation='relu')(dense1)

input2 = Input(shape=(3,))
dense2 = Dense(10, activation='relu')(input2)
dense2 = Dense(20, activation='relu')(dense2)
dense2 = Dense(50, activation='relu')(dense2)
dense2 = Dense(5, activation='relu')(dense2)


#모델병합 / concatenate:연쇄시키다, 연관시키다
from tensorflow.keras.layers import concatenate, Concatenate
# from keras.layers import concatenate, Concatenate
# from keras.layers.merge import concatenate, Concatenate

#병합모델의 layer구성
merge1 = concatenate([dense1, dense2]) 
middle1 = Dense(30) (merge1) 
middle1 = Dense(30) (middle1)
middle1 = Dense(30) (middle1)
middle1 = Dense(30) (middle1)
 
#모델분기 1
output1 = Dense(30)(middle1)
output1 = Dense(10)(output1)
output1 = Dense(3)(output1) 

#모델분기 2
output2 = Dense(30)(middle1)
output2 = Dense(10)(output2)
output2 = Dense(10)(output2)
output2 = Dense(3)(output2) 

#모델선언
model = Model(inputs=[input1, input2], outputs=[output1, output2]) 

model.summary()

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae','mse'])
model.fit([x1_train,x2_train], [y1_train, y2_train], epochs=50, batch_size=1, validation_split=0.2, verbose=1) 

#4. 평가, 예측
loss = model.evaluate([x1_test,x2_test], [y1_test, y2_test], batch_size=1)
print('model.metrics_names : ', model.metrics_names)
print('loss: ', loss)

y1_predict, y2_predict = model.predict([x1_test,x2_test])

# RMSE, R2
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse1 = RMSE(y1_test, y1_predict)
rmse2 = RMSE(y2_test, y2_predict)
rmse = (rmse1+rmse2) / 2
print('RMSE1: ',rmse1)
print('RMSE2: ',rmse2)
print('RMSE: ',rmse)


from sklearn.metrics import r2_score
r2_1=r2_score(y1_test, y1_predict)
r2_2=r2_score(y2_test, y2_predict)
r2=(r2_1+r2_2)/2
print('R2_1: ',r2_1)
print('R2_2: ',r2_2)
print('R2: ',r2)