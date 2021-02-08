# epochs = 2, validation_split = 0.2 추가
# RandomizedSearchCV
# KerasClassifier

# callbacks
# ReduceLROnPlateau 3
# EarlyStopping 5


import numpy as np 
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 1. 데이터/ 전처리
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape)
print(y_test.shape)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

# 2. 모델
def build_model(drop=0.5,optimizer='adam'):
    inputs = Input(shape = (28*28),name = 'input')
    x = Dense(512,activation='relu',name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256,activation='relu',name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128,activation='relu',name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10,activation='softmax',name = 'outputs')(x)
    model = Model(inputs=inputs,outputs=outputs)
    model.compile(loss = 'categorical_crossentropy',optimizer = optimizer,metrics = ['acc'])
    return model

def create_hyperparameter() : 
    batchs = [50, 60, 70]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    return {'batch_size' : batchs, 'optimizer' : optimizers, 'drop':dropout} 

def callbacks():
    modelpath ='../data/modelcheckpoint/k61_4_{epoch:2d}_{val_loss:.4f}.hdf5'
    er = EarlyStopping(monitor = 'val_loss',patience=5)
    mo = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath,save_best_only=True)
    lr = ReduceLROnPlateau(monitor = 'val_loss',patience=3)
    return er,mo,lr

er,mo,lr = callbacks()  

hyperparameters = create_hyperparameter()
model = build_model()

#===========================================================================================래핑 :  KerasClassifier
# TypeError: If no scoring is specified, the estimator passed should have a 'score' method. 
# The estimator <tensorflow.python.keras.engine.functional.Functional object at 0x000001AE964FBE80> does not.
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose = 1)   #, epochs = 2)

search = RandomizedSearchCV(model2, hyperparameters, cv=3)
# search = GridSearchCV(model2, hyperparameters, cv=3)

search.fit(x_train, y_train, verbose=1, epochs = 100, validation_split = 0.2, callbacks = [er, lr, mo])

acc = search.score(x_test, y_test) 
print(search.best_params_)      # 최적의 파라미터 값 출력
print(search.best_estimator_)   
print(search.best_score_)       # 최고의 점수
print('최종스코어 : ', acc)     # 최종스코어 :  0.9638000130653381

'''
=========================================KerasClassifier 사용전 : 
mnist_DNN : 
[0.28995245695114136, 0.9696999788284302]
y_pred[:10] :  [7 2 1 0 4 1 4 9 5 9]
y_test[:10] :  [7 2 1 0 4 1 4 9 5 9]

mnist_CNN : 
[0.15593186020851135, 0.9835000038146973]
y_pred[:10] :  [7 2 1 0 4 1 4 9 5 9]
y_test[:10] :  [7 2 1 0 4 1 4 9 5 9]

mnist_LSTM :
[0.1378639042377472, 0.9803000092506409]
y_pred[:10] :  [7 2 1 0 4 1 4 9 5 9]
y_test[:10] :  [7 2 1 0 4 1 4 9 5 9]


=========================================KerasClassifier 사용후 : 
dnn : 
{'optimizer': 'adam', 'drop': 0.1, 'batch_size': 10}
<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001E008507AF0>
0.9576666752497355
최종스코어 :  0.9638000130653381

cnn:                                                                                        ***
{'optimizer': <class 'tensorflow.python.keras.optimizer_v2.adam.Adam'>, 'learning_rate': 0.001, 'kernel_size': 3, 'filters': 100, 'drop': 0.1, 'batch_size': 70}
<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000002D70711B910>
0.9784666697184244
최종스코어 :  0.9860000014305115

lstm:
{'optimizer': 'adam', 'drop': 0.3, 'batch_size': 40}
<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0000029407203400>
0.9333833257357279
최종스코어 :  0.9503999948501587

=====================================# epochs = 2, validation_split = 0.2, callbacks[ReduceLROnPlateau 3,EarlyStopping 5, modelcheckpoint ]
fit epochs = 100 : 
{'optimizer': 'adam', 'drop': 0.1, 'batch_size': 50}
<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x00000182232AED30>
0.9800666570663452
최종스코어 :  0.9860000014305115

KerasClassifier epochs = 100 :
{'optimizer': 'adam', 'drop': 0.2, 'batch_size': 50}
<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x00000243C3FE14C0>
0.9807999928792318
최종스코어 :  0.9853000044822693
'''



