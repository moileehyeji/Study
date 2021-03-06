# pip install autokeras

import numpy as np
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.datasets import mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)/255
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)/255

# onehot해보자
# onehot안하면 y_train.shape = (60000,) ---> 돌아감
# onehot하면 y_train.shape = (60000,10) ---> 돌아감=====> 결론: 해도되고 안해도 됨
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#=======================================
model = ak.ImageClassifier(
                            overwrite=True, 
                            max_trials=10,      # 최대시도:2
                            loss = 'mse',
                            metrics=['acc']     # 기본값 : accuracy
)

# model.summary()   #모델이 완성되기 전이므로 결과 안나옴

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

es =EarlyStopping(monitor='val_loss', mode='min', patience=6)
re = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose = 2)
mo = ModelCheckpoint('../data/keras3/modelcheckpoint/', save_weights_only=True, save_best_only=True, monitor='val_loss', verbose=1)

model.fit(x_train, y_train, epochs=10, validation_split=0.2, callbacks = [es, re, mo]) # validation_splitr 기본 0.2

results = model.evaluate(x_test, y_test)

print(results)  #[0.058278825134038925, 0.9807999730110168]

model.summary()   #AttributeError: 'ImageClassifier' object has no attribute 'summary'
