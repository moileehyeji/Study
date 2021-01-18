#다:1 mlp 함수형
#keras10_mlp2을 함수형으로 바꾸시오


#1. 데이터
import numpy as np

#input3 -> output1
#행렬표현법2
x = np.array([range(100), range(301,401), range(1,101)]) #[[0~100], [301~400], [1~100]] 
y = np.array(range(711,811))

print(x.shape) #(3,100)
print(y.shape) #(100,)

x=np.transpose(x)

# print(x)
# print(x.shape) #(100,3)

#train, test 행 자르기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66) 



#2. 모델구성
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Input

input1=Input(shape=(3,))
dense=Dense(5, activation='relu')(input1)
dense=Dense(3)(dense)
dense=Dense(40)(dense)
output1=Dense(1)(dense)
model=Model(inputs=input1, outputs=output1)


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
x = np.array([range(100), range(301,401), range(1,101)]) #[[0~100], [301~400], [1~100]] 
y = np.array(range(711,811))
'''

x_predict2=np.array([100,401,101])
x_predict2=x_predict2.reshape(1,3)
y_predict2=model.predict(x_predict2)
print('y_predict2: ',y_predict2)

''' 
loss:  2.9802322831784522e-09
mae:  3.662109520519152e-05
RMSE:  5.459150335692846e-05
R2:  0.9999999999962291
y_predict2:  [[811.]] 
'''