import numpy as np 
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist, boston_housing
from sklearn.datasets import load_diabetes
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# 1. 데이터/ 전처리
dataset = load_diabetes() 

x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size = 0.2, shuffle = True, random_state = 66)


print(x_train.shape)   
print(y_test.shape)     

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
def build_model(drop=0.5,optimizer='adam'):
    inputs = Input(shape = (x_train.shape[1],),name = 'input')
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
    modelpath ='../data/modelcheckpoint/k62_2_{epoch:2d}_{val_loss:.4f}.hdf5'
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
DNN모델 : 
loss :  2166.6650390625 
mae :   38.769779205322266
RMSE :  46.54745007951129
R2 :    0.6297812819678937

LSTM모델 : 
loss :  2578.437255859375 
mae :   40.43071746826172
rmse :  50.778316251131514
r2 :    0.5594216266685046

CNN모델 : 
loss :  3345.784423828125
mae :  46.672889709472656
rmse :  57.842756300367
r2 :  0.48447401699219805


=========================================KerasClassifier 사용후 : 

=====================================# epochs = 2, validation_split = 0.2, callbacks[ReduceLROnPlateau 3,EarlyStopping 5, modelcheckpoint ]
DNN모델 : 
{'optimizer': 'rmsprop', 'drop': 0.2, 'batch_size': 50}
<tensorflow.python.keras.wrappers.scikit_learn.KerasRegressor object at 0x0000022E704B60A0>
-2938.1461588541665
최종스코어 :  -3315.5439453125
r2_score :  0.4891335432063365

'''
