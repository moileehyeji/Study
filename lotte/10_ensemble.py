import numpy as np
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation, Dropout
from tensorflow.keras.layers import concatenate
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import VGG19, MobileNet, ResNet101, ResNet50
from tensorflow.keras.optimizers import Adam


''' 
#=========================================데이터 지정 및 전처리
x = np.load("C:/Study/lotte/data/npy/128_project_x.npy",allow_pickle=True)
x_pred = np.load('C:/Study/lotte/data/npy/128_test.npy',allow_pickle=True)
y = np.load("C:/Study/lotte/data/npy/128_project_y.npy",allow_pickle=True)

# print(x.shape, x_pred.shape, y.shape)   #(48000, 128, 128, 3) (72000, 128, 128, 3) (48000, 1000)

x = preprocess_input(x) # (48000, 255, 255, 3)
x_pred = preprocess_input(x_pred)   # 

x_train, x_valid, y_train, y_valid = train_test_split(x,y, train_size = 0.9, shuffle = True, random_state=66)


#=========================================IDG
idg = ImageDataGenerator(
    # rotation_range=10, acc 하락
    width_shift_range=(-1,1),  
    height_shift_range=(-1,1), 
    rotation_range=40, 
    # shear_range=0.2)    # 현상유지
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
idg2 = ImageDataGenerator()

train_generator = idg.flow(x_train,y_train,batch_size=50000, seed = 2048)
# seed => random_state
valid_generator = idg2.flow(x_valid,y_valid, batch_size=72000)
# test_generator = idg2.flow(x_pred)

x_train = train_generator[0][0]
y_train = train_generator[0][1]
x_valid = valid_generator[0][0]
y_valid = valid_generator[0][1]

np.save("C:/Study/lotte/data/npy/10_128_x_train_IDG.npy", arr = x_train)
np.save("C:/Study/lotte/data/npy/10_128_y_train_IDG.npy", arr = y_train)
np.save("C:/Study/lotte/data/npy/10_128_x_valid_IDG.npy", arr = x_valid)
np.save("C:/Study/lotte/data/npy/10_128_y_valid_IDG.npy", arr = y_valid)

print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
# (43200, 128, 128, 3) (43200, 1000)
# (4800, 128, 128, 3) (4800, 1000) 
# '''

x_train = np.load("C:/Study/lotte/data/npy/10_128_x_train_IDG.npy",allow_pickle=True)
x_valid = np.load('C:/Study/lotte/data/npy/10_128_x_valid_IDG.npy',allow_pickle=True)
y_train = np.load("C:/Study/lotte/data/npy/10_128_y_train_IDG.npy",allow_pickle=True)
y_valid = np.load("C:/Study/lotte/data/npy/10_128_y_valid_IDG.npy",allow_pickle=True)
x_pred = np.load('C:/Study/lotte/data/npy/128_test.npy',allow_pickle=True)

# print(x_train.shape, y_train.shape)
# print(x_valid.shape, y_valid.shape)
# (43200, 128, 128, 3) (43200, 1000)
# (4800, 128, 128, 3) (4800, 1000)

#=========================================모델 앙상블
# ValueError: The name "conv1_bn" is used 2 times in the model. All layer names should be unique.
# [해결]for l in res.layers:
#       l._name = "%s_workaround" % l.name
# INPUT
res = ResNet50(include_top=False,weights='imagenet',input_shape=x_train.shape[1:])
for l in res.layers:
      l._name = "%s_workaround" % l.name
res.trainable = True  
a = res.output
a = GlobalAveragePooling2D() (a)
a = Flatten() (a)
a = Dense(4048, activation= 'relu') (a)
a = Dropout(0.2) (a)
# a = Dense(1000, activation= 'softmax') (a) 

eff = MobileNet(weights="imagenet", include_top=False, input_shape=x_train.shape[1:])
for l in eff.layers:
      l._name = "%s_workaround2" % l.name
eff.trainable = True
b = eff.output
b = GlobalAveragePooling2D()(b)
b = Flatten()(b)
b = Dense(2024, activation= 'relu')(b)
b = Dropout(0.2) (b)

# 앙상블
merge = concatenate([a,b])
merge1 = Dense(1024, activation="relu")(merge)

#OUTPUT
merge1 = Dense(1000, activation="softmax")(merge1)
model = Model(inputs=[res.input, eff.input], outputs = merge1)


#=========================================컴파일, 훈련
mc = ModelCheckpoint('C:/Study/lotte/data/h5/10_ensemble.hdf5',save_best_only=True, verbose=1)
early_stopping = EarlyStopping(patience= 20)
lr = ReduceLROnPlateau(patience= 10, factor=0.5)

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-5), metrics=['acc'])
learning_history = model.fit([x_train, x_train], y_train,epochs=1,
    validation_data=([x_valid, x_valid], y_valid), callbacks=[early_stopping,lr,mc])



#=========================================predict
model.load_weights('C:/Study/lotte/data/h5/10_ensemble.hdf5')
result = model.predict(x_pred,verbose=True)



#==========================================제출생성
sub = pd.read_csv('C:/Study/lotte/data/sample.csv')
sub['prediction'] = np.argmax(result,axis = 1)
sub.to_csv('C:/Study/lotte/data/10.csv',index=False)

  