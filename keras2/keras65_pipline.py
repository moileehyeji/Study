# pipeline 사용


import numpy as np 
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 1. 데이터/ 전처리
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape)
print(y_test.shape)

x_train = x_train.reshape(60000, 28*28) #.astype('float32')/255.
x_test = x_test.reshape(10000, 28*28)   #.astype('float32')/255.

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
    batchs = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    # return {'mo__batch_size' : batchs, 'mo__optimizer' : optimizers, 'mo__drop':dropout}  
    return {'kerasclassifier__batch_size' : batchs, 'kerasclassifier__optimizer' : optimizers, 'kerasclassifier__drop':dropout}  

    

hyperparameters = create_hyperparameter()
model = build_model()

#===========================================================================================래핑 :  KerasClassifier
# TypeError: If no scoring is specified, the estimator passed should have a 'score' method. 
# The estimator <tensorflow.python.keras.engine.functional.Functional object at 0x000001AE964FBE80> does not.
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose = 1)

# pipe = Pipeline([('scaler', MinMaxScaler()), ('mo', model2)])
pipe = make_pipeline(MinMaxScaler(), model2)

search = RandomizedSearchCV(pipe, hyperparameters, cv=3)
# search = GridSearchCV(model2, hyperparameters, cv=3)


search.fit(x_train, y_train)

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

=========================================Pipeline 사용후 : 
{'mo__optimizer': 'rmsprop', 'mo__drop': 0.3, 'mo__batch_size': 40}
Pipeline(steps=[('scaler', MinMaxScaler()),
                ('mo',
                 <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0000022A43F01EB0>)])
0.955216666062673
최종스코어 :  0.953000009059906

'''



