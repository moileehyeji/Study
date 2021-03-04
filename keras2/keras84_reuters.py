# 짧은 뉴스 기사와 토픽의 집합인 로이터 데이터셋
# 46개의 토픽
# 각 토픽은 훈련 세트에 최소한 10개의 샘플
from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Nadam, Adam, RMSprop
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#reuters
#num_words: 위에서부터 단어 10000개 가져오겠다.
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=5000, #단어 빈도에 따라 유지할 최대 단어 수입니다.
                                                    test_split=0.2)      
                                                    

print(x_train[0], type(x_train[0]))   #<class 'list'>
print(y_train[0], type(y_train[0]))   #3 <class 'numpy.int64'>
print(len(x_train[0]), len(x_train[11]))    #87 59
print('-------------------------------------------------')
print(x_train.shape, x_test.shape)  #(8982,) (2246,)
print(y_train.shape, y_test.shape)  #(8982,) (2246,)

print('뉴스기사의 최대길이 : ', max(len(l) for l in x_train))   #2376
print('뉴스기사의 평균길이 : ', sum(map(len, x_train)) / len(x_train))  #145.5398574927633

''' #시각화
#막대그래프
plt.hist([len(s) for s in x_train], bins=50)
plt.show()  # 한쪽에 치우쳐있음 WHY? 가장 많이 나오는 단어가 첫번째임 '''


#y분포
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print('y_분포 : ', dict(zip(unique_elements, counts_elements)))
# y_train category : 46개 
print('-------------------------------------------------')


''' #시각화
plt.hist(y_train, bins=46)
plt.show()  #0~5에 치우쳐있음 '''

#x의 단어분포
word_to_index = reuters.get_word_index()
print(word_to_index)
print(len(word_to_index))   #30979  --> word_size즉 input_dim
print(type(word_to_index))  #<class 'dict'>
print('-------------------------------------------------')


#시퀀스화 되어있는 데이터를 문장화
#키와 벨류를 교체
index_to_word = {}  #dintionary
for key, value in word_to_index.items():
    index_to_word[value] = key
#교환 후
print(index_to_word)
print(index_to_word[1])
print(index_to_word[30979])
print(len(index_to_word))   #30979


#x_train[0]
print(x_train[0])
print(' '.join([index_to_word[index] for index in x_train[0]]))


#y카테고리 개수 출력
category = np.max(y_train)+1
print('y카테고리 개수 : ', category)    #46

#y의 유니크한 값 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)

############################전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, padding='pre', maxlen=500, truncating='pre')
x_test = pad_sequences(x_test, padding='pre', maxlen=500, truncating='pre')
print(x_train.shape, x_test.shape)  #(8982, 500), (2246, 500) 


#loss='sparse_categorical_crossentropy'
''' y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape)  #(8982, 46) (2246, 46) '''

# 2. 모델구성
from tensorflow.keras.layers import Embedding, LSTM, Dense, Flatten, Conv1D, Dropout
from tensorflow.keras.models import Sequential
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=500))
model.add(LSTM(64, activation='tanh'))
# model.add(Conv1D(64,2, activation='tanh'))
# model.add(Flatten())
# model.add(Dense(32, activation='tanh'))
# model.add(Dense(64, activation='tanh'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='tanh'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='tanh'))
model.add(Dense(46, activation='softmax'))

model.summary()

re = ReduceLROnPlateau(monitor='val_loss', patience=15, factor=0.5, verbose=1, mode='auto')
er = EarlyStopping(monitor='val_loss', patience=30, mode='auto')

#sparse_categorical_crossentropy(다중 분류 손실함수)
#y값을 원핫인코딩한것돠 동일한 효과를 볼 수 있음
# model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01), metrics=['acc'])
model.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01), metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=30, validation_split=0.2, callbacks=[er])

loss, acc = model.evaluate(x_test, y_test, batch_size=30)
print('loss:', loss)        
print('acc:', acc)        


