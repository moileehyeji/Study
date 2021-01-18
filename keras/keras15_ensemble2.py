#실습: 다:1 앙상블 구현

#1. 데이터
import numpy as np

x1 = np.array([range(100), range(301,401), range(1,101)])
x2 = np.array([range(101,201), range(411,511), range(100,200)])
y1 = np.array([range(711,811), range(1,101), range(201,301)])

x1=np.transpose(x1)
x2=np.transpose(x2)
y1=np.transpose(y1)

from sklearn.model_selection import train_test_split

# train_test_split 3가지 데이터
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(x1, x2, y1, train_size=0.8, shuffle=True)


#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense

#모델1
input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu')(input1)
dense1 = Dense(30, activation='relu')(dense1)
dense1 = Dense(140, activation='relu')(dense1)
dense1 = Dense(50, activation='relu')(dense1)

#모델2
input2 = Input(shape=(3,))
dense2 = Dense(10, activation='relu')(input2)
dense2 = Dense(130, activation='relu')(dense2)
dense2 = Dense(70, activation='relu')(dense2)
dense2 = Dense(90, activation='relu')(dense2) 


#모델병합 / concatenate:연쇄시키다, 연관시키다
from tensorflow.keras.layers import concatenate, Concatenate

#병합모델의 layer구성
merge1 = concatenate([dense1, dense2]) 
middle1 = Dense(35) (merge1) 
middle1 = Dense(50) (middle1)
middle1 = Dense(50) (middle1)
middle1 = Dense(60) (middle1)
middle1 = Dense(80) (middle1)
 
#모델분기 
output1 = Dense(30)(middle1)
output1 = Dense(60)(output1)
output1 = Dense(100)(output1)
output1 = Dense(20)(output1)
output1 = Dense(3)(output1) #최종 아웃풋

#모델선언
model = Model(inputs=[input1, input2], outputs=output1) 

model.summary()

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train,x2_train], y1_train, epochs=100, batch_size=1, validation_split=0.2, verbose=1) 

#4. 평가, 예측
loss = model.evaluate([x1_test,x2_test], y1_test, batch_size=1)
print('model.metrics_names : ', model.metrics_names)
print('loss: ', loss)

y_predict = model.predict([x1_test,x2_test])

print('========================================================')
print('y1_predict : \n', y_predict)
print('========================================================')

# RMSE, R2
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE: ',RMSE(y1_test, y_predict))


from sklearn.metrics import r2_score
print('R2: ',r2_score(y1_test, y_predict))