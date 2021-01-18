#다:1 mlp

#1. 데이터
import numpy as np

#input2 -> output1
#행렬표현법1
x = np.array([ [1,2,3,4,5,6,7,8,9,10] , [11,12,13,14,15,16,17,18,19,20] ])
y = np.array([1,2,3,4,5,6,7,8,9,10])

print(x.shape) #(2,10)

#실습 x행렬(2,10) -> (10,2)로 변환
'''
x = np.arange(20).reshape(10,-1)
x= np.reshape(x,(10,2))
[[ 1  2]
 [ 3  4]
 [ 5  6]
 [ 7  8]
 [ 9 10]
 [11 12]
 [13 14]
 [15 16]
 [17 18]
 [19 20]]
(10, 2)
'''
x=np.transpose(x)
'''
[[ 1 11]
 [ 2 12]
 [ 3 13]
 [ 4 14]
 [ 5 15]
 [ 6 16]
 [ 7 17]
 [ 8 18]
 [ 9 19]
 [10 20]]
(10, 2)
'''


print(x)
print(x.shape) #(10,2)                                                                                                                                                                                                                                                                                                         

#input_dim=1 ->1차원 -> vector

'''
#실습
#열우선 (가장 작은 단위부터) [ [ [4,5,6] , [1,2,3] ] ] ->(1,2,3)

print(np.array([[1,2,3],[4,5,6]]).shape) #(2,3)
print(np.array([[1,2],[3,4],[5,6]]).shape) #(3,2)
print(np.array([[[1,2,3],[4,5,6]]]).shape) #(1,2,3)
print(np.array([[1,2,3,4,5,6]]).shape) #(1,6)
print(np.array([[[1,2],[4,5]],[[5,6],[7,8]]]).shape) #(2,2,2)
print(np.array([[1],[2],[3]]).shape) #(3,1)
print(np.array([[[1],[2]],[[3],[4]]]).shape) #(2,2,1)

'''

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.layers import Dense # 가능하지만 속도가 느려짐

model= Sequential()
model.add(Dense(10,input_dim=2))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam',metrics=['mae'])
model.fit(x,y,epochs=100, batch_size=1, validation_split=0.2) # 각 컬럼별로 2개

#4. 평가, 예측
loss, mae=model.evaluate(x,y)
print('loss: ',loss)
print('mae: ',mae)

#loss:  3.45877921859028e-08
#mae:  0.00016033649444580078

y_predict=model.predict(x)
#print(y_predict)

'''
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print('RMSE: ',RMSE(y_test,y_predict))
#print('mae: ',mean_squared_error(y_test,y_predict))
print('mae: ',mean_squared_error(y_predict,y_test))


from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print('R2: ',r2)
'''








