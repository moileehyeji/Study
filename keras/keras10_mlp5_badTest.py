#다:다 mlp

#실습
#1. R2          : 0.5이하/음수는 안돼
#2. layer       : 5개 이상
#3. node        : 각 10개 이상
#4. batch_size  : 8 이하
#5. epochs      : 30 이상

'''
1. epochs=30, batch_size=7
R2:  0.24770053854994967
2.
R2:  0.37008078475475686
3.
R2:  0.9069859429389786
'''
'''
1.epochs=40, batch_size=7
R2:  0.1904022909953056
2.
R2:  0.7249547793946092
3.
R2:  0.9824074425412952
'''
'''
1.epochs=40, batch_size=1
R2:  0.9999999999930147
'''

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
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(2))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=30, batch_size=7, validation_split=0.2) #x_val:(16,5),y_val:(16,2)

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
#print('mae: ',mean_squared_error(y_test,y_predict))
#print('mae: ',mean_squared_error(y_predict,y_test))


from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print('R2: ',r2)

'''
x_pred2=np.array([[100,402,101,901,601]]) #(5,)
#x_pred2=np.transpose(x_pred2) #(5,)
x_pred2=x_pred2.reshape(1,5) #(1,5)
print('x_pred2 shape: ',x_pred2.shape)
y_pred2=model.predict(x_pred2)
print(y_pred2)
print('y_pred2 shape: ',y_pred2.shape)

'''