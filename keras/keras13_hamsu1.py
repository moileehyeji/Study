
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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
# from keras.layers import Dense

#함수형 모델 : Input,Dense1(input),Dense2(dense1),Dense3(dense2)...outputs(dense3)
input1 = Input(shape=(5,)) #input layer 직접 구성
dense1 = Dense(5, activation='relu')(input1)
dense2 = Dense(3)(dense1)
dense3 = Dense(4)(dense2)
outputs = Dense(2)(dense3)
model = Model(inputs = input1, outputs = outputs)
model.summary() # 모델구성출력

'''
#함수형 모델 : Input,Dense1(input),Dense2(dense1),Dense3(dense2)...outputs(dense3)
input1 = Input(shape=(1,)) #input layer 직접 구성
dense1 = Dense(5, activation='relu')(input1)
dense2 = Dense(3)(dense1)
dense3 = Dense(4)(dense2)
outputs = Dense(1)(dense3)
model = Model(inputs = input1, outputs = outputs)
model.summary()

Model: "functional_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 1)]               0
_________________________________________________________________
dense (Dense)                (None, 5)                 10
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 18
_________________________________________________________________
dense_2 (Dense)              (None, 4)                 16
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 5
=================================================================
Total params: 49
Trainable params: 49
Non-trainable params: 0
_________________________________________________________________
'''
'''
#순차적 모델 
model= Sequential()
model.add(Dense(5, activation='relu', input_shape=(1,)))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))
model.summary()

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_4 (Dense)              (None, 5)                 10
_________________________________________________________________
dense_5 (Dense)              (None, 3)                 18
_________________________________________________________________
dense_6 (Dense)              (None, 4)                 16
_________________________________________________________________
dense_7 (Dense)              (None, 1)                 5
=================================================================
Total params: 49
Trainable params: 49
Non-trainable params: 0
_________________________________________________________________
'''




#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=50000, batch_size=1, validation_split=0.2, verbose=3) 


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

