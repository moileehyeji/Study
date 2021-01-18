#다:다 앙상블 구현

#1. 데이터
import numpy as np

x1 = np.array([range(100), range(301,401), range(1,101)])
x2 = np.array([range(101,201), range(411,511), range(100,200)])

y1 = np.array([range(711,811), range(1,101), range(201,301)])
y2 = np.array([range(501,601), range(711,811), range(100)])
y3 = np.array([range(601,701), range(811,911), range(1100,1200)])

x1=np.transpose(x1)
x2=np.transpose(x2)
y1=np.transpose(y1)
y2=np.transpose(y2)
y3=np.transpose(y3)

from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=0.8, shuffle=True)
x2_train, x2_test, y2_train, y2_test, y3_train, y3_test = train_test_split(x2, y2, y3, train_size=0.8, shuffle=True)


#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense

#모델1
input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu')(input1)
dense1 = Dense(30, activation='relu')(dense1)
dense1 = Dense(30, activation='relu')(dense1)
dense1 = Dense(40, activation='relu')(dense1)
dense1 = Dense(50, activation='relu')(dense1)

#모델2
input2 = Input(shape=(3,))
dense2 = Dense(10, activation='relu')(input2)
dense2 = Dense(30, activation='relu')(dense2)
dense2 = Dense(30, activation='relu')(dense2)
dense2 = Dense(70, activation='relu')(dense2)
dense2 = Dense(90, activation='relu')(dense2) 



#모델병합 / concatenate:연쇄시키다, 연관시키다
from tensorflow.keras.layers import concatenate, Concatenate

#병합모델의 layer구성
merge1 = concatenate([dense1, dense2]) #model1의 끝과 model2의 끝 concatnate
middle1 = Dense(35) (merge1) 
middle1 = Dense(50) (middle1)
middle1 = Dense(50) (middle1)
middle1 = Dense(60) (middle1)
middle1 = Dense(80) (middle1)
 
 
#모델분기 1
output1 = Dense(30)(middle1)
output1 = Dense(10)(output1)
output1 = Dense(3)(output1) #최종 아웃풋

#모델분기 2
output2 = Dense(30)(middle1)
output2 = Dense(10)(output2)
output2 = Dense(10)(output2)
output2 = Dense(3)(output2) #최종 아웃풋

#모델분기 3
output3 = Dense(30)(middle1)
output3 = Dense(50)(output3)
output3 = Dense(70)(output3)
output3 = Dense(70)(output3)
output3 = Dense(3)(output3) #최종 아웃풋

#모델선언
model = Model(inputs=[input1, input2], outputs=[output1, output2, output3]) 

model.summary()


#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train,x2_train], [y1_train, y2_train, y3_train], epochs=100, batch_size=1, validation_split=0.2, verbose=3) #x,y각각 두개->리스트

#4. 평가, 예측
loss = model.evaluate([x1_test,x2_test], [y1_test, y2_test, y3_test], batch_size=1)
print('model.metrics_names : ', model.metrics_names)
print('loss: ', loss)

y1_predict, y2_predict, y3_predict = model.predict([x1_test,x2_test]) # y값 두가지 반환

""" print('========================================================')
print('y1_predict : \n', y1_predict)
print('y2_predict : \n', y2_predict)
print('y3_predict : \n', y3_predict)
print('========================================================')
 """
# RMSE, R2
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse1 = RMSE(y1_test, y1_predict)
rmse2 = RMSE(y2_test, y2_predict)
rmse3 = RMSE(y3_test, y3_predict)
rmse = (rmse1+rmse2+rmse3) / 3
# print('RMSE1: ',rmse1)
# print('RMSE2: ',rmse2)
# print('RMSE3: ',rmse3)
print('RMSE: ',rmse)


from sklearn.metrics import r2_score
r2_1=r2_score(y1_test, y1_predict)
r2_2=r2_score(y2_test, y2_predict)
r2_3=r2_score(y3_test, y3_predict)
r2=(r2_1+r2_2+r2_3)/3
# print('R2_1: ',r2_1)
# print('R2_2: ',r2_2)
# print('R2_3: ',r2_3)
print('R2: ',r2)


""" 
x1 = np.array([range(100), range(301,401), range(1,101)])
x2 = np.array([range(101,201), range(411,511), range(100,200)])

y1 = np.array([range(711,811), range(1,101), range(201,301)])
y2 = np.array([range(501,601), range(711,811), range(100)])
y3 = np.array([range(601,701), range(811,911), range(1100,1200)]) """

x1_predict = np.array([100,401,101])
x2_predict = np.array([201,511,200])
x1_predict=x1_predict.reshape(1,3)
x2_predict=x2_predict.reshape(1,3)

y_predict = model.predict([x1_predict,x2_predict])
print('y_predict: ', y_predict)


