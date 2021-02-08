import numpy as np 
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist, boston_housing
from sklearn.datasets import load_diabetes, load_breast_cancer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,  accuracy_score


# 1. 데이터/ 전처리
dataset = load_breast_cancer() 

x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size = 0.2, shuffle = True, random_state = 66)


print(x_train.shape)   
print(y_test.shape)     

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# 2. 모델
def build_model(drop=0.5,optimizer='adam'):
    inputs = Input(shape = (x_train.shape[1],),name = 'input')
    x = Dense(512,activation='relu',name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256,activation='relu',name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128,activation='relu',name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(1,activation='sigmoid',name = 'outputs')(x)
    model = Model(inputs=inputs,outputs=outputs)
    model.compile(loss = 'binary_crossentropy',optimizer = optimizer,metrics = ['acc'])
    return model

def create_hyperparameter() : 
    batchs = [50, 60, 70]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    return {'batch_size' : batchs, 'optimizer' : optimizers, 'drop':dropout} 

def callbacks():
    modelpath ='../data/modelcheckpoint/k62_3_{epoch:2d}_{val_loss:.4f}.hdf5'
    er = EarlyStopping(monitor = 'val_loss',patience=5)
    mo = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath,save_best_only=True)
    lr = ReduceLROnPlateau(monitor = 'val_loss',patience=3)
    return er,mo,lr

er,mo,lr = callbacks()  

hyperparameters = create_hyperparameter()

model2 = KerasClassifier(build_fn=build_model, verbose = 1)   #, epochs = 2)

search = RandomizedSearchCV(model2, hyperparameters, cv=3)
# search = GridSearchCV(model2, hyperparameters, cv=3)

search.fit(x_train, y_train, verbose=1, epochs = 100, validation_split = 0.2, callbacks = [er, lr, mo])

acc = search.score(x_test, y_test) 
print(search.best_params_)      # 최적의 파라미터 값 출력
print(search.best_estimator_)   
print(search.best_score_)       # 최고의 점수
print('최종스코어 : ', acc)     # 최종스코어 : 

'''
=========================================KerasClassifier 사용전 : 
DNN모델 : 
loss :  0.22489522397518158
acc :  0.9736841917037964
y_pred[:10] :  [[1 1 0 1 1 1 1 1 1 1 0 1 0 1 1 0 1 0 0 0]]
y_test[:10] :  [1 1 0 0 1 1 1 1 1 1 0 1 0 1 1 0 1 0 0 0]

LSTM모델 : 
loss :  0.3034729063510895
acc :  0.9385964870452881
y_pred[:10] :  [[1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 0 1 0 0 0]]
y_test[:10] :  [1 1 0 0 1 1 1 1 1 1 0 1 0 1 1 0 1 0 0 0]

CNN모델 : 
loss :  1.6061084270477295
acc :  0.9122806787490845
y_pred[:10] :  [[1 1 1 1 1 1 0 1 1 1]]
y_test[:10] :  [1 1 1 1 1 0 0 1 1 1]


=========================================KerasClassifier 사용후 : 

=====================================# epochs = 2, validation_split = 0.2, callbacks[ReduceLROnPlateau 3,EarlyStopping 5, modelcheckpoint ]
DNN모델 : 
{'optimizer': 'adam', 'drop': 0.3, 'batch_size': 70}
<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0000020088099FD0>
0.9714476664861044
최종스코어 :  0.9649122953414917

'''
