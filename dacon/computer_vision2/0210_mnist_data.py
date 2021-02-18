import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
warnings.filterwarnings("ignore")

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Nadam
from tensorflow.python.keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

# 1. 데이터

x_train = np.load('./dacon/computer2/data/npy/0210_mnist_data_x_train.npy')
x_test = np.load('./dacon/computer2/data/npy/0210_mnist_data_x_test.npy')
x_val = np.load('./dacon/computer2/data/npy/0210_mnist_data_x_val.npy')
x_pred = np.load('./dacon/computer2/data/npy/0210_mnist_data_x_pred.npy')
y_train = np.load('./dacon/computer2/data/npy/0210_mnist_data_y_train.npy')
y_test = np.load('./dacon/computer2/data/npy/0210_mnist_data_y_test.npy')
y_val = np.load('./dacon/computer2/data/npy/0210_mnist_data_y_val.npy')


# ImageDatagenerator & data augmentation
idg = ImageDataGenerator(height_shift_range=(-1,1),width_shift_range=(-1,1))
idg2 = ImageDataGenerator()
train_generator = idg.flow(x_train,y_train,batch_size=8)
test_generator = idg2.flow(x_test,y_test)
val_generator = idg2.flow(x_val,y_val)
pred_generator = idg2.flow(x_pred,shuffle=False)

# 2, 3. 모델, 훈련
model = load_model('./dacon/computer2/data/h5/[0.91]0210_mnist_data.h5')

# 4. 평가, 예측
loss = model.evaluate(test_generator, batch_size=8)
print('loss : ', loss)

# submit
submit = pd.read_csv('./dacon/computer2/data/mnist_data/submission.csv', header=0)

y_pred = model.predict_generator(pred_generator)
y_pred = np.argmax(y_pred, axis=1)
submit.loc[:,'digit'] = y_pred

submit.to_csv('./dacon/computer2/data/csv/0210_mnist_data.csv',index=False)



'''

def build_model():
    model = Sequential()
        
    model.add(Conv2D(16,(3,3),activation='relu',input_shape=(28,28,1),padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
        
    model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='relu',padding='same')) 
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3,3)))
    model.add(Dropout(0.3))
        
    model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(5,5),activation='relu',padding='same')) 
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3,3)))
    model.add(Dropout(0.3))
        
    model.add(Flatten())

    model.add(Dense(128,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(10,activation='softmax'))

    opti = Nadam(learning_rate=0.001) 

    # 3. 컴파일   
    model.compile(loss='categorical_crossentropy', optimizer=opti, metrics='acc')
    return model

def callbacks():
    modelpath = './dacon/computer2/data/modelcheckpoint/0210_mnist_data_{epoch:02d}_{val_acc:.4f}.hdf5'
    er = EarlyStopping(monitor='val_acc', patience=20, mode= 'auto')
    lr = ReduceLROnPlateau(monitor='val_acc', patience=5, factor=0.6, verbose=1) 
    mo = ModelCheckpoint(filepath=modelpath, monitor='val_acc', save_best_only=True, mode='auto')
    return er,mo,lr


# 1. 데이터
train = pd.read_csv('./dacon/computer2/data/mnist_data/train.csv', header=0)
test = pd.read_csv('./dacon/computer2/data/mnist_data/test.csv', header=0)
submit = pd.read_csv('./dacon/computer2/data/mnist_data/submission.csv', header=0)

# print(train)
# print(train.shape)  # (2048, 787)
# print(test.shape)   # (20480, 786)

# drop columns
train2 = train.drop(['id','digit','letter'], axis = 1)
test2 = test.drop(['id','letter'], axis = 1)

# print(df_x.shape)      # (2048, 786)
# print(df_y.shape)      # (2048,)

x = train2.iloc[:,:] 
y = train.loc[:,'digit']

x = x.to_numpy()
y = y.to_numpy()
x_pred = test2.to_numpy()


# 데이터 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state=104)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, shuffle = True, random_state=104)

x_train = x_train/255.
x_test = x_test/255.
x_val = x_val/255.
x_pred = x_pred/255.

print(x_train.shape, x_test.shape)  # (1638, 784) (410, 784)
print(y_train.shape, y_test.shape)  # (1638,) (410,)
print(x_pred.shape)                 # (20480, 784)

x_train = x_train.reshape(-1, 28, 28, 1)       
x_test = x_test.reshape(-1,28, 28, 1)     
x_val = x_val.reshape(-1, 28, 28, 1)  
x_pred = x_pred.reshape(-1, 28, 28, 1)   

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)


np.save('./dacon/computer2/data/npy/0210_mnist_data_x_train.npy', arr=x_train)
np.save('./dacon/computer2/data/npy/0210_mnist_data_x_test.npy', arr=x_test)
np.save('./dacon/computer2/data/npy/0210_mnist_data_x_val.npy', arr=x_val)
np.save('./dacon/computer2/data/npy/0210_mnist_data_x_pred.npy', arr=x_pred)
np.save('./dacon/computer2/data/npy/0210_mnist_data_y_train.npy', arr=y_train)
np.save('./dacon/computer2/data/npy/0210_mnist_data_y_test.npy', arr=y_test)
np.save('./dacon/computer2/data/npy/0210_mnist_data_y_val.npy', arr=y_val)


# ImageDatagenerator & data augmentation
idg = ImageDataGenerator(height_shift_range=(-1,1),width_shift_range=(-1,1))
idg2 = ImageDataGenerator()
train_generator = idg.flow(x_train,y_train,batch_size=8)
test_generator = idg2.flow(x_test,y_test)
val_generator = idg2.flow(x_val,y_val)
pred_generator = idg2.flow(x_pred,shuffle=False)




# 2. 모델
model = build_model()

# 3. 훈련
er,mo,lr = callbacks()
model.fit_generator(train_generator, epochs=1000,validation_data=val_generator , callbacks=[er,mo,lr])

model.save('./dacon/computer2/data/h5/0210_mnist_data.h5')
# model = load_model('./dacon/computer2/data/h5/[0.90]0210_mnist_data.h5')

# 4. 평가, 예측
loss = model.evaluate(test_generator, batch_size=8)
print('loss : ', loss)


# submit
y_pred = model.predict_generator(pred_generator)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)
print(y_pred.shape)

submit.loc[:,'digit'] = y_pred

submit.to_csv('./dacon/computer2/data/csv/0210_mnist_data.csv',index=False)
'''

"""   
loss :  [0.3034801483154297, 0.9073171019554138]

"""

