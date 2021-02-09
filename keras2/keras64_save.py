# 가중치 저장
# 1. model.save사용
# 2. pickle사용


import numpy as np 
import warnings
warnings.filterwarnings('ignore')
import pickle
import joblib

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 1. 데이터/ 전처리
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape)
print(y_test.shape)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.
'''
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
    return {'batch_size' : batchs, 'optimizer' : optimizers, 'drop':dropout}   

hyperparameters = create_hyperparameter()
model = build_model()

#===========================================================================================래핑 :  KerasClassifier
# TypeError: If no scoring is specified, the estimator passed should have a 'score' method. 
# The estimator <tensorflow.python.keras.engine.functional.Functional object at 0x000001AE964FBE80> does not.
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose = 1)

search = RandomizedSearchCV(model2, hyperparameters, cv=3)
# search = GridSearchCV(model2, hyperparameters, cv=3)


search.fit(x_train, y_train, verbose=1)

acc = search.score(x_test, y_test) 
print(search.best_params_)      # 최적의 파라미터 값 출력
print(search.best_estimator_)   
print(search.best_score_)       # 최고의 점수
print('최종스코어 : ', acc)     # 최종스코어 :  0.9638000130653381



# 모델저장 : 모두 저장은 되지만 load는 model.save만 됨
search.best_estimator_.model.save('../data/h5/k64_Random_model.save.h5') 
pickle.dump(search.best_estimator_.model, open('../data/h5/k64_Random_model.save.pickle.dat', 'wb'))        
pickle.dump(model2, open('../data/h5/k64_Random_model2.save.pickle.dat', 'wb'))  
        
# search.best_estimator_.model.save('../data/h5/k64_Grid_model.save.h5') 
# # pickle.dump(search.best_estimator_, open('../data/h5/k64_Grid_model2.save.pickle.dat', 'wb'))     
# pickle.dump(model2, open('../data/h5/k64_Grid_model2.save.pickle.dat', 'wb'))                                  
  
print('저장완료')

'''
# 모델 불러오기
# pickle_model = pickle.load(open('../data/h5/k64_Random_model2.save.pickle.dat', 'rb'))    #안됨
# -> AttributeError :  'build_model'속성을 가져올 수 없습니다.
# pickle_model = pickle.load(open('../data/h5/k64_Random_model.save.pickle.dat', 'rb'))     #안됨
model3 = load_model('../data/h5/k64_Random_model.save.h5')                                  #됨
print('불러왔다')

# 평가
# acc = pickle_model.score(x_test, y_test)
y_pred = model3.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
print('score : ',accuracy_score(y_test, y_pred))    # score :  0.9678
print(y_pred.shape) # (10000,)


'''

'''



