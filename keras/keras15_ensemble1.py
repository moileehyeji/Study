#다:다 앙상블 구현

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

#모델1
input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu')(input1)
dense1 = Dense(5, activation='relu')(dense1 )#model1의 끝
# output1 = Dense(3)(dense1)

#모델2
input2 = Input(shape=(3,))
dense2 = Dense(10, activation='relu')(input2)
dense2 = Dense(10, activation='relu')(dense2)
dense2 = Dense(10, activation='relu')(dense2)
dense2 = Dense(10, activation='relu')(dense2) #model2의 끝
# output2 = Dense(3)(dense2)


#모델병합 / concatenate:연쇄시키다, 연관시키다
from tensorflow.keras.layers import concatenate, Concatenate
# from keras.layers import concatenate, Concatenate
# from keras.layers.merge import concatenate, Concatenate

#병합모델의 layer구성
merge1 = concatenate([dense1, dense2]) #model1의 끝과 model2의 끝 concatnate
middle1 = Dense(30) (merge1) 
middle1 = Dense(30) (middle1)
middle1 = Dense(30) (middle1)
middle1 = Dense(30) (middle1)
 
#모델분기 1
output1 = Dense(30)(middle1)
output1 = Dense(10)(output1)
output1 = Dense(3)(output1) #최종 아웃풋

#모델분기 2
output2 = Dense(30)(middle1)
output2 = Dense(10)(output2)
output2 = Dense(10)(output2)
output2 = Dense(3)(output2) #최종 아웃풋

#모델선언
model = Model(inputs=[input1, input2], outputs=[output1, output2]) #두개이상은 list로 표시

model.summary()

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train,x2_train], [y1_train, y2_train], epochs=50, batch_size=1, validation_split=0.2, verbose=1) #x,y각각 두개->리스트

#4. 평가, 예측
loss = model.evaluate([x1_test,x2_test], [y1_test, y2_test], batch_size=1)
print('model.metrics_names : ', model.metrics_names)
print('loss: ', loss)

""" 
loss 5개

        total loss         1st모델loss(mse)   2rd모델loss(mse)    1st모델 metrics(mse)2rd모델metrics(mse)
loss:  [975.3253173828125, 473.2955627441406, 502.02972412109375, 473.2955627441406, 502.02972412109375]


975.3253173828125 = 473.2955627441406 + 502.0297241210937

"""

""" 
x1 = np.array([range(100), range(301,401), range(1,101)])
y1 = np.array([range(711,811), range(1,101), range(201,301)])

x2 = np.array([range(101,201), range(411,511), range(100,200)])
y2 = np.array([range(501,601), range(711,811), range(100)])
 """

y1_predict, y2_predict = model.predict([x1_test,x2_test]) # y값 두가지 반환

print('========================================================')
print('y1_predict : \n', y1_predict)

print('y2_predict : \n', y2_predict)
print('========================================================')

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