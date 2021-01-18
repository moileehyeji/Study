import numpy as np

x_data = np.load('../data/npy/iris_x.npy')
y_data = np.load('../data/npy/iris_y.npy')

print(x_data)
print(y_data)
print(x_data.shape)
print(y_data.shape)

#모델을 완성하시오.

# 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size = 0.8, random_state = 120, shuffle = True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.6, random_state = 120, shuffle = True)

# x 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

# y 전처리(sklearn)
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)


#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(30, input_shape = (4,), activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
""" model.add(Dense(50, activation='relu'))
model.add(Dense(90, activation='relu'))
model.add(Dense(70, activation='relu')) """
model.add(Dense(3, activation='softmax')) #output레이어 노드 : 분류하고자 하는 숫자의 갯수, y 컬럼

#3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early = EarlyStopping(monitor = 'acc', patience=20, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=50, validation_data=(x_val, y_val), callbacks=[early])

#4.평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss, acc :', loss)

y_pre = model.predict(x_test[:10])
# print('y_pre : \n', y_pre)
print('y_pre2 : \n', y_pre)
print('y실제값 \n: ', y_test[:10])


#결과치 나오게 코딩할 것 : argmax
y_pre = np.argmax(y_pre, axis=1)
print('y_pre : \n', y_pre)


'''
Dense모델 : 
loss, acc : [0.054814912378787994, 1.0]

LSTM모델 sklearn : 
loss, acc : [0.007020933087915182, 1.0]

load npy : 
loss, acc : [0.06003233417868614, 1.0]
'''
