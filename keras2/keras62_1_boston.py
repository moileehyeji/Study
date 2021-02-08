import numpy as np 
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist, boston_housing
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# 1. 데이터/ 전처리
print(x_train.shape)    # (404, 13)
print(y_test.shape)     # (102, 51)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
def build_model(drop=0.5,optimizer='adam'):
    inputs = Input(shape = (13,),name = 'input')
    x = Dense(512,activation='relu',name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256,activation='relu',name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128,activation='relu',name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(1,name = 'outputs')(x)
    model = Model(inputs=inputs,outputs=outputs)
    model.compile(loss = 'mse',optimizer = optimizer,metrics = ['mae'])
    return model

def create_hyperparameter() : 
    batchs = [50, 60, 70]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    return {'batch_size' : batchs, 'optimizer' : optimizers, 'drop':dropout}  

def callbacks():
    modelpath ='../data/modelcheckpoint/k62_1_{epoch:2d}_{val_loss:.4f}.hdf5'
    er = EarlyStopping(monitor = 'val_loss',patience=5)
    mo = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath,save_best_only=True)
    lr = ReduceLROnPlateau(monitor = 'val_loss',patience=3)
    return er,mo,lr

er,mo,lr = callbacks() 

hyperparameters = create_hyperparameter()

model2 = KerasRegressor(build_fn=build_model, verbose = 1)   #, epochs = 2)

search = RandomizedSearchCV(model2, hyperparameters, cv=3)
# search = GridSearchCV(model2, hyperparameters, cv=3)

search.fit(x_train, y_train, verbose=1, epochs = 100, validation_split = 0.2, callbacks = [er, lr, mo])

y_pred = search.best_estimator_.model.predict(x_test)
r2 = search.score(x_test, y_test) 
print(search.best_params_)      # 최적의 파라미터 값 출력
print(search.best_estimator_)   
print(search.best_score_)       # 최고의 점수
print('최종스코어 : ', r2)     # 최종스코어 :  0.9638000130653381
print('r2_score : ', r2_score(y_test, y_pred))

'''
=========================================KerasClassifier 사용전 : 
Dense모델 : 
loss, mae :  2166.6650390625 38.769779205322266
RMSE :  46.54745007951129
R2 :  0.6297812819678937

LSTM모델 : 
loss, mae :  2578.437255859375 40.43071746826172
rmse :  50.778316251131514
r2 :  0.5594216266685046

Conv1D
loss, mae :  2085.394287109375 36.618099212646484
rmse :  45.66612357134248
r2 :  0.6436679568820876

Conv2D  : 
loss :  4.494823455810547
mae :  1.6760056018829346
rmse :  2.120099468937249
r2 :  0.9462232137123261


=========================================KerasClassifier 사용후 : 

=====================================# epochs = 2, validation_split = 0.2, callbacks[ReduceLROnPlateau 3,EarlyStopping 5, modelcheckpoint ]
Dense모델 : 
{'optimizer': 'adam', 'drop': 0.1, 'batch_size': 60}
<tensorflow.python.keras.wrappers.scikit_learn.KerasRegressor object at 0x00000145FE8D41C0>
-15.026178677876791
최종스코어 :  -28.877403259277344
r2_score :  0.6530986310965947

'''
