
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
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(30, input_shape = (30,), activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#hidden이 없는 모델 가능

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
path = '../data/modelcheckpoint/k45_cancer_{epoch:02d}_{val_loss:.4f}.hdf5'
mc = ModelCheckpoint(filepath=path, monitor='val_loss', save_best_only=True, mode = 'auto')
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=100, validation_data = (x_val, y_val), callbacks=[early_stopping, mc])

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

#시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')
plt.grid()

plt.title('Loss Cost')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc = 'upper right')

plt.subplot(2,1,2)
plt.plot(hist.history['acc'], marker = '.', c = 'red', label = 'acc')
plt.plot(hist.history['val_acc'], marker = '.', c = 'blue', label = 'val_acc')
plt.grid()

plt.title('Accuracy')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(loc = 'upper right')

plt.show()

'''
Dense모델 : 
loss :  0.22489522397518158
acc :  0.9736841917037964
y값 :  [1 1 0 0 1 1 1 1 1 1 0 1 0 1 1 0 1 0 0 0]
[[1 1 0 1 1 1 1 1 1 1 0 1 0 1 1 0 1 0 0 0]]

'''

