
#1. 데이터
import numpy as np

#행렬표현법2
x = np.array([range(100), range(301,401), range(1,101), range(801,901), range(501,601)])
y = np.array([range(711,811), range(1,101)])
print(x.shape) #(5,100)
print(y.shape) #(2,100)

x=np.transpose(x)
y=np.transpose(y)
print(x.shape) #(100,5)
print(y.shape) #(100,2)

#train, test 행 자르기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66) 
print(x_train.shape)    #(80,5)
print(y_train.shape)    #(80,2)


#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.layers import Dense

model= Sequential()
model.add(Dense(10,input_dim=5))
model.add(Dense(101))
model.add(Dense(34))
model.add(Dense(67))
model.add(Dense(2))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=100, batch_size=1, validation_split=0.2, verbose=9) 

'''
# 평가에서도 사용
verbose = 0 : 훈련과정 표시 X 
verbose = 1 : Epoch 100/100
                64/64 [==============================] - 0s 1ms/step - loss: 2.1506e-08 - mae: 9.5800e-05 - val_loss: 1.1999e-07 - val_mae: 2.4992e-04
                1/1 [==============================] - 0s 0s/step - loss: 1.1535e-07 - mae: 2.4686e-04
verbose = 2 : Epoch 100/100
                64/64 - 0s - loss: 0.0013 - mae: 0.0255 - val_loss: 6.2602e-05 - val_mae: 0.0072
                1/1 [==============================] - 0s 0s/step - loss: 5.6079e-05 - mae: 0.0067
verbose = 3 : Epoch 100/100
                1/1 [==============================] - 0s 0s/step - loss: 133.1006 - mae: 11.1673

'''

#4. 평가, 예측
loss, mae=model.evaluate(x_test,y_test)
print('loss: ',loss)
print('mae: ',mae)

y_predict=model.predict(x_test)
#print(y_predict)


from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict): #shape가 같아야 함
    return np.sqrt(mean_squared_error(y_test,y_predict))
print('RMSE: ',RMSE(y_test,y_predict))

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print('R2: ',r2)

x_pred2=np.array([[100,402,101,901,601]]) #(5,)
#x_pred2=np.transpose(x_pred2) #(5,)
x_pred2=x_pred2.reshape(1,5) #(1,5)
print('x_pred2 shape: ',x_pred2.shape)
y_pred2=model.predict(x_pred2)
print(y_pred2)
print('y_pred2 shape: ',y_pred2.shape)

