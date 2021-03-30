import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from keras import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam,SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB4,EfficientNetB2,EfficientNetB1
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation, Conv2D, Dropout
from tensorflow.keras import regularizers




#데이터 지정 및 전처리
x = np.load("C:/data/lotte/npy/128_project_x.npy",allow_pickle=True)
x_pred = np.load('C:/data/lotte/npy/128_test.npy',allow_pickle=True)
y = np.load("C:/data/lotte/npy/128_project_y.npy",allow_pickle=True)


x = preprocess_input(x)
x_pred = preprocess_input(x_pred)

idg = ImageDataGenerator(
    width_shift_range=(-1,1),  
    height_shift_range=(-1,1), 
    rotation_range=45, 
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

idg2 = ImageDataGenerator()


x_train, x_valid, y_train, y_valid = train_test_split(x,y, train_size = 0.9, shuffle = True, random_state=66)

train_generator = idg.flow(x_train,y_train,batch_size=32, seed = 42)
valid_generator = idg2.flow(x_valid,y_valid)



efficientnet = EfficientNetB2(include_top=False,weights='imagenet',input_shape=x_train.shape[1:])
a = efficientnet.output
a = Conv2D(filters = 16,kernel_size=(12,12), strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(1e-5)) (a)
a = BatchNormalization() (a)
a = Activation('swish') (a)
a = Conv2D(filters = 16,kernel_size=(12,12), strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(1e-5)) (a)
a = BatchNormalization() (a)
a = Activation('swish') (a)
a = GlobalAveragePooling2D() (a)
a = Dense(256, activation= 'swish') (a)
a = Dropout(0.5) (a)
a = Dense(1000, activation= 'softmax') (a)

model = Model(inputs = efficientnet.input, outputs = a)

# efficientnet.summary()

mc = ModelCheckpoint('C:/data/lotte/h5/[{val_loss:.4f}]33_effi_cnn2.h5',save_best_only=True, verbose=1)
early_stopping = EarlyStopping(patience= 10)
lr = ReduceLROnPlateau(patience= 5, factor=0.4)

model.compile(loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.02, momentum=0.9),
            metrics=['acc'])
# learning_history = model.fit_generator (train_generator,epochs=100, steps_per_epoch= len(x_train) / 32,
#     validation_data=valid_generator, callbacks=[early_stopping,lr,mc])

# predict
model.load_weights('C:/data/lotte/h5/[0.0154]33_effi_cnn2.h5')
result = model.predict(x_pred,verbose=True)

sub = pd.read_csv('C:/data/lotte/csv/sample.csv')
sub['prediction'] = np.argmax(result,axis = 1)
sub.to_csv('C:/data/lotte/csv/33_2.csv',index=False)

# [0.0103]33_effi_cnn2.h5/33    : 75.974
# [0.0125]33_effi_cnn2.h5/33_1  : 75.524
# [0.0154]33_effi_cnn2.h5/33_2  : 74.843