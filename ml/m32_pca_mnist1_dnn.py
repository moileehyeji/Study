# m31로 만든 0.95이상의 n_conponent = ?를 사용하여 
# dnn모델을 만들것

# mnist dnn 보다 성능 좋게 만들어라!!
# cnn과 비교!!


# keras40_mnist3_dnn 복사

#1. 데이터
import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)    #(60000, 28, 28)
print(x_test.shape)     #(10000, 28, 28)
print(y_train.shape)    #(60000,)
print(y_test.shape)     #(10000,)

# PCA
from sklearn.decomposition import PCA

x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)

print(x.shape)      #(70000, 28, 28)

# ValueError: Found array with dim 3. Estimator expected <= 2.
x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])

'''
pca = PCA()
pca.fit(x)

cumsum = np.cumsum(pca.explained_variance_ratio_)
print('cumsum : ', cumsum)

d = np.argmax(cumsum >= 0.95) + 1
print('cumsum >= 0.95  :', cumsum >= 0.95)
print('선택할 차원의 수 :', d)              # 선택할 차원의 수 : 154

'''

pca = PCA(n_components=154)
x = pca.fit_transform(x)

print(x.shape)  #(70000, 154)
print(y.shape)  #(70000,)


#전처리 
from sklearn.model_selection import train_test_split
x_train , x_test, y_train, y_test  = train_test_split(x,y, test_size = 0.2, shuffle=True, random_state = 66)


from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(20, input_shape=(154,), activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early = EarlyStopping(monitor='loss', patience=20, mode= 'auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=500, batch_size=200, callbacks=[early])

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print(loss)


x_pre = x_test[:10]
y_pre = model.predict(x_pre)
y_pre = np.argmax(y_pre, axis=1)
y_test_pre = np.argmax(y_test[:10], axis=1)
print('y_pred[:10] : ', y_pre)
print('y_test[:10] : ', y_test_pre)

'''
mnist_CNN : 
[0.15593186020851135, 0.9835000038146973]
y_pred[:10] :  [7 2 1 0 4 1 4 9 5 9]
y_test[:10] :  [7 2 1 0 4 1 4 9 5 9]

mnist_DNN : 
[0.28995245695114136, 0.9696999788284302]
y_pred[:10] :  [7 2 1 0 4 1 4 9 5 9]
y_test[:10] :  [7 2 1 0 4 1 4 9 5 9]

pca_mnist_DNN (cumsum >= 0.95): 
[0.4257657527923584, 0.9498571157455444]
y_pred[:10] :  [0 4 8 0 5 9 8 2 1 0]
y_test[:10] :  [0 4 8 0 5 9 8 2 1 0]
'''
