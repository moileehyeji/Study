#다:다 mlp

#1. 데이터
import numpy as np

#input3 -> output3
#행렬표현법2
x = np.array([range(100), range(301,401), range(1,101)])
y = np.array([range(711,811), range(1,101), range(201,301)])
print(x.shape) #(3,100)
print(y.shape) #(3,100)

x=np.transpose(x)
y=np.transpose(y)
print(x.shape) #(100,3)
print(y.shape) #(100,3)


#train, test 행 자르기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66) 
print(x_train.shape)    #(80,3)
print(y_train.shape)    #(80,3)


#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.layers import Dense

model= Sequential()
model.add(Dense(10,input_dim=3))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(3)) #output_dim=3

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=100, batch_size=1, validation_split=0.2) #batch_size=1 -> (1,3)

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
