# 실습: 드립아웃적용

import numpy as np
from sklearn.datasets import load_breast_cancer

# 1. 데이터
dataset = load_breast_cancer()

print(dataset.DESCR)  
print(dataset.feature_names)  
""" 
:Number of Instances: 569

:Number of Attributes: 30 numeric, predictive attributes and the class 
    
--> (569,30)
"""

x = dataset.data
y = dataset.target

print(x.shape)  #(569, 30)
print(y.shape)  #(569,)
#--->데이터셋은 31 컬럼구성
print(x[:5])
print(y)    #0 , 1 두가지로 569개 출력

# 전처리 : minmax, train_test_split

#전처리 :y전처리는 다중분류일때 적용
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test  =  train_test_split(x, y, train_size=0.8, random_state = 120)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.6, random_state = 120, shuffle = True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(30, input_shape = (30,), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(60, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
#hidden이 없는 모델 가능

# 3. 컴파일, 훈련
# loss='binary_crossentropy'
#                   'mean_squared_error'    --> 풀네임 가능
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, validation_data = (x_val, y_val))

loss, acc = model.evaluate(x_test,y_test)
print('loss : ', loss)
print('acc : ', acc)

#이진분류 0,1 출력
y_pre = model.predict(x_test[:20])
y_pre = np.transpose(y_pre)
# print('y_pre : ', y_pre)
print('y값 : ', y_test[:20])

y_pre = np.where(y_pre<0.5, 0, 1)
# y_pre = np.argmax(y_pre, axis=1)
print(y_pre)

""" 
<출력>
loss :  0.2202349305152893
acc :  0.9649122953414917

y_pre :  [[1 1 0 1 1 1 1 1 1 1 0 1 0 1 1 0 1 0 0 0]]
y값 :  [1 1 0 0 1 1 1 1 1 1 0 1 0 1 1 0 1 0 0 0]
[0]
 """

""" 
잘못된 평가지표 : 
loss :  0.09340619295835495
acc :  0.10526315867900848  

수정된 평가지표 : 
*binaye_crossentropy, sigmoid
loss :  0.1538175344467163
acc :  0.9473684430122375
---> 성능이 향상된 것이 아니라 평가지표를 맞게 수정한 것

실습 1. acc 0.98이상:
1. loss :  0.06174115091562271
acc :  0.9824561476707458

실습 2. y값 출력해볼것(y[-5:-1](y전체 -1:가장 끝값, -2: 가장 끝에서 두번째 ), predict 비교해볼것):
결과값: 소수점
loss :  0.05343734100461006
acc :  0.9912280440330505
y_pre : 
 [[0.]
 [0.]
 [0.]
 [0.]] 

 --> 0<n<1값으로 나오기 때문에 0,1 둘중에 출력되지 않음
"""

'''
Dense모델 : 
loss :  0.22489522397518158
acc :  0.9736841917037964
y값 :  [1 1 0 0 1 1 1 1 1 1 0 1 0 1 1 0 1 0 0 0]
[[1 1 0 1 1 1 1 1 1 1 0 1 0 1 1 0 1 0 0 0]]

LSTM모델 : 
loss :  0.3300505578517914
mae :  0.06349937617778778
y값 :  [1 1 0 0 1 1 1 1 1 1 0 1 0 1 1 0 1 0 0 0]
[[1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 0 1 0 0 0]]

dropout 후:
loss :  0.1321868747472763
acc :  0.9736841917037964
y값 :  [1 1 0 0 1 1 1 1 1 1 0 1 0 1 1 0 1 0 0 0]
[[1 1 0 1 1 1 1 1 1 1 0 1 0 1 1 0 1 0 0 0]]

'''
