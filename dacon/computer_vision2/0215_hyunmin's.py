import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import PIL.Image as pilimg
from PIL import Image



######################################################
# File Load
train = pd.read_csv('./dacon/computer2/data/dirty_mnist_2nd_answer.csv')
print(train.shape)  # (50000, 27)

sub = pd.read_csv('./dacon/computer2/data/sample_submission.csv')
print(sub.shape)    # (5000, 27)

######################################################
'''
#1. DATA

#### train
df_x = []

for i in range(0,50000):
    if i < 10 :
        file_path = './dacon/computer2/data/dirty_mnist_2nd/0000' + str(i) + '.png'
    elif i >=10 and i < 100 :
        file_path = './dacon/computer2/data/dirty_mnist_2nd/000' + str(i) + '.png'
    elif i >= 100 and i <1000 :
        file_path = './dacon/computer2/data/dirty_mnist_2nd/00' + str(i) + '.png'
    elif i >= 1000 and i < 10000 :
        file_path = './dacon/computer2/data/dirty_mnist_2nd/0' + str(i) + '.png'
    else : 
        file_path = './dacon/computer2/data/dirty_mnist_2nd/' + str(i) + '.png'
    image = pilimg.open(file_path)
    # image = image.resize((64,64))
    image = image.resize((50,50))
    pix = np.array(image)
    pix = pd.DataFrame(pix)
    df_x.append(pix)

x = pd.concat(df_x)
x = x.values
# print("x.shape ", x.shape)       # (12800000, 256) >>> (50000, 50, 50, 1)
# print(x[0,:])
x[100 < x] = 253
x[x < 100] = 0
x = x.reshape(50000, 50, 50, 1)
print("x.shape ", x.shape)      # (50000, 50, 50, 1)

y = train.iloc[:,1:]
y = y.values
print("y.shape ", y.shape)    # (50000, 26)

np.save('./dacon/computer2/data/npy/vision_50_x3.npy', arr=x)
np.save('./dacon/computer2/data/npy/vision_50_y3.npy', arr=y)

#### pred
df_pred = []

for i in range(0,5000):
    if i < 10 :
        file_path = './dacon/computer2/data/test_dirty_mnist_2nd/5000' + str(i) + '.png'
    elif i >=10 and i < 100 :
        file_path = './dacon/computer2/data/test_dirty_mnist_2nd/500' + str(i) + '.png'
    elif i >= 100 and i <1000 :
        file_path = './dacon/computer2/data/test_dirty_mnist_2nd/50' + str(i) + '.png'
    else : 
        file_path = './dacon/computer2/data/test_dirty_mnist_2nd/5' + str(i) + '.png'
    image = pilimg.open(file_path)
    # image = image.resize((64, 64))
    image = image.resize((50,50))
    pix = np.array(image)
    pix = pd.DataFrame(pix)
    df_pred.append(pix)

x_pred = pd.concat(df_pred)
x_pred = x_pred.values
print(x_pred.shape)       # (1280000, 256) >>> (5000, 50, 50, 1)

x_pred = x_pred.reshape(5000, 50, 50, 1)
x_pred[100 < x_pred] = 253
x_pred[x_pred < 100] = 0
print("x_pred.shape ", x_pred.shape)       # (5000, 50, 50, 1)
np.save('./dacon/computer2/data/npy//vision_50_x_pred3.npy', arr=x_pred)

'''

#1. DATA Load
x = np.load('./dacon/computer2/data/npy/vision_50_x3.npy')
y = np.load('./dacon/computer2/data/npy/vision_50_y3.npy')
x_pred = np.load('./dacon/computer2/data/npy//vision_50_x_pred3.npy')
print("<==complete load==>")

print(x.shape, y.shape, x_pred.shape) # (50000, 50, 50, 1) (50000, 26) (5000, 50, 50, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=47)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=47)
print(x_train.shape, x_test.shape, x_valid.shape)  # (32000, 50, 50, 1) (10000, 50, 50, 1) (8000, 50, 50, 1)
print(y_train.shape, y_test.shape, y_valid.shape)  # (32000, 26) (10000, 26) (8000, 26)


train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.2,
    fill_mode='nearest'
)
etc_datagen = ImageDataGenerator(rescale=1./255)

batch = 16

train_generator = train_datagen.flow(x_train, y_train, batch_size=batch, seed=2021)
test_generator = etc_datagen.flow(x_test, y_test, batch_size=batch, seed=2021)
valid_generator = etc_datagen.flow(x_valid, y_valid)
pred_generator = etc_datagen.flow(x_pred)

#2. Modeling
model = Sequential()
model.add(Conv2D(32, (2,2), padding='same', input_shape=(50, 50, 1), activation='relu'))
model.add(BatchNormalization()) 
model.add(Conv2D(32, (2,2), padding='same', activation='relu'))
model.add(BatchNormalization()) 
model.add(AveragePooling2D(2,2))

model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
model.add(BatchNormalization()) 
model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
model.add(BatchNormalization()) 
model.add(AveragePooling2D(2,2))

model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
model.add(BatchNormalization()) 
model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
model.add(BatchNormalization()) 
model.add(AveragePooling2D(2,2))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization()) 
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization()) 
model.add(Dense(26, activation='softmax'))

#3. Compile, Train
es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.4, mode='min')

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01, epsilon=None), metrics=['acc'])
hist = model.fit_generator(train_generator, epochs=50, \
    steps_per_epoch = len(x_train) // batch , validation_data=valid_generator, callbacks=[es, lr])

#4. Evaluate, Predict
loss, acc = model.evaluate(test_generator)
print("loss : ", loss)
print("acc : ", acc)  

# loss :  214.3434295654297
# acc :  0.017899999395012856

y_pred = model.predict_generator(pred_generator)
y_pred[y_pred<0.5] = 0
y_pred[y_pred>=0.5] = 1
print(y_pred.shape) # (5000, 26)

sub.iloc[:,1:] = y_pred

sub.to_csv('./dacon/computer2/data/csv/0215_hyunmin.csv', index=False)
print(sub.head())

# 0215_hyunmin.csv
# score 0.53447



