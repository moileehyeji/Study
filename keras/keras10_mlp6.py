#1:다 mlp

#1. 데이터
import numpy as np

#input1 -> output3 (권장모델은 아님, 가능함을 보여줌)
x = np.array(range(100))#(100,)
y = np.array([range(711,811), range(1,101), range(201,301)])
print(x.shape) 
print(y.shape) 

# x=np.transpose(x)
x=x.reshape(100,1)
y=np.transpose(y)
print(x.shape) 
print(y.shape) 


#train, test 행 자르기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66) 
print(x_train.shape)    
print(y_train.shape)    


#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.layers import Dense

model= Sequential()
model.add(Dense(10,input_dim=1))
model.add(Dense(53))
model.add(Dense(65))
model.add(Dense(101))
model.add(Dense(3)) #output_dim=3

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=100, batch_size=1, validation_split=0.2, verbose=3) #batch_size=1 -> (1,3)

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
x = np.array(range(100))#(100,)
y = np.array([range(711,811), range(1,101), range(201,301)]) 
'''

x_predict2=np.array([100])
y_predict2=model.predict(x_predict2)
print('y_predict2: ', y_predict2)

''' 
1.
loss:  1.8816412872268984e-09
mae:  2.7870137273566797e-05
RMSE:  4.3377889893767536e-05
R2:  0.9999999999976191
y_predict2:  [[811.00006 101.      301.00006]] 
2.
loss:  1.3378220753423875e-09
mae:  1.9800663721980527e-05
RMSE:  3.657625157070095e-05
R2:  0.9999999999983072
y_predict2:  [[811.0002  100.99999 301.00003]]
3. 
loss:  1.9050818877985876e-07
mae:  0.00025919676409102976
RMSE:  0.00043647245387391114
R2:  0.9999999997589503
y_predict2:  [[811.00146 100.99999 301.00052]]
'''
 