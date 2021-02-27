import torch
import glob
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F
from sklearn.model_selection import KFold
import time
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
from edafa import ClassPredictor
from torch_poly_lr_decay import PolynomialLRDecay
import random
import albumentations

torch.set_num_threads(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ======================== Dacon Dataset Load ==========================
# Dacon Dataset Load
labels_df = pd.read_csv('C:/Study/dacon/computer2/data/dirty_mnist_2nd_answer.csv')[:]
imgs_dir = np.array(sorted(glob.glob('C:/Study/dacon/computer2/data/dirty_mnist_2nd/*')))[:]
labels = np.array(labels_df.values[:,1:])
test_imgs_dir = np.array(sorted(glob.glob('C:/Study/dacon/computer2/data/test_dirty_mnist_2nd/*')))

imgs=[]
for path in tqdm(imgs_dir[:]):
    img=cv2.imread(path, cv2.IMREAD_COLOR)
    imgs.append(img)
imgs=np.array(imgs)

# 저장소에서 load
class MnistDataset_v1(Dataset):
    def __init__(self, imgs_dir=None, labels=None, transform=None, train=True):
        self.imgs_dir = imgs_dir
        self.labels = labels
        self.transform = transform
        self.train = train
        pass
    
    def __len__(self):
        # 데이터 총 샘플 수
        return len(self.imgs)
    
    def __getitem__(self, idx):
        # 1개 샘플 get
        img = cv2.imread(self.imgs_dir[idx], cv2.IMREAD_COLOR)
        img = self.transform(img)
        if self.train==True:
            label = self.labels[idx]
            return img, label
        else:
            return img
        
        pass
    
# 메모리에서 load
class MnistDataset_v2(Dataset):
    def __init__(self, imgs=None, labels=None, transform=None, train=True):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.train=train
                #test data augmentations
        self.aug = albumentations.Compose ([ 
                   albumentations.RandomResizedCrop (256, 256), 
                    albumentations.Transpose (p = 0.5), 
                    albumentations.HorizontalFlip (p = 0.5), 
                    albumentations.VerticalFlip (p = 0.5)
                    ], p = 1) 
        pass
    
    def __len__(self):
        # 데이터 총 샘플 수
        return len(self.imgs)
    
    def __getitem__(self, idx):
        # 1개 샘플 get1
        img = self.imgs[idx]
        img = self.transform(img)
        
        if self.train==True:
            label = self.labels[idx]
            return img, label
        else:
            return img

# ========================= reproduction을 위한 seed 설정=====================
# https://dacon.io/competitions/official/235697/codeshare/2363?page=1&dtype=recent&ptype=pub
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  

class myPredictor(ClassPredictor):
    def __init__(self,model,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.model = model

    def predict_patches(self,patches):
        return self.model.predict(patches)


# ==================== model 정의 ==================================
# EfficientNet -b0(pretrained)
# MultiLabel output

class EfficientNet_MultiLabel(nn.Module):
    def __init__(self, in_channels):
        super(EfficientNet_MultiLabel, self).__init__()
        self.network = EfficientNet.from_pretrained('efficientnet-b7', in_channels=in_channels) # b3, b7 
        self.output_layer = nn.Linear(1000, 26)

    def forward(self, x):
        x = F.relu(self.network(x))
        x = torch.sigmoid(self.output_layer(x))
        return x

# ============== 데이터 분리====================================
# 해당 코드에서는 1fold만 실행

kf = KFold(n_splits=5, shuffle=True, random_state=42)
folds=[]
for train_idx, valid_idx in kf.split(imgs):
    folds.append((train_idx, valid_idx))

### seed_everything(42)

# 5개의 fold 모두 실행하려면 for문을 5번 돌리면 됩니다.
for fold in range(1):
    model = EfficientNet_MultiLabel(in_channels=3).to(device)
# model = nn.DataParallel(model)
    train_idx = folds[fold][0]
    valid_idx = folds[fold][1]

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
        ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        ])

    epochs=25
    batch_size=5        # 자신의 VRAM에 맞게 조절해야 OOM을 피할 수 있습니다.
    
    # Data Loader
    train_dataset = MnistDataset_v2(imgs = imgs[train_idx], labels=labels[train_idx], transform=train_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = MnistDataset_v2(imgs = imgs[valid_idx], labels = labels[valid_idx], transform=valid_transform)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False) 
    
    # optimizer
    # polynomial optimizer를 사용합니다.
    # 
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    decay_steps = (len(train_dataset)//batch_size)*epochs
    scheduler_poly_lr_decay = PolynomialLRDecay(optimizer, max_decay_steps=decay_steps, end_learning_rate=1e-5, power=0.9)

    criterion = torch.nn.BCELoss()
    
    epoch_accuracy = []
    valid_accuracy = []
    valid_losses=[]
    valid_best_accuracy=0
    for epoch in range(epochs):
        model.train()
        batch_accuracy_list = []
        batch_loss_list = []
        start=time.time()
        for n, (X, y) in enumerate((train_loader)):
            X = torch.tensor(X, device=device, dtype=torch.float32)
            y = torch.tensor(y, device=device, dtype=torch.float32)
            y_hat = model(X)
                        
            optimizer.zero_grad()
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            scheduler_poly_lr_decay.step()
            
            y_hat  = y_hat.cpu().detach().numpy()
            y_hat = y_hat>0.5
            y = y.cpu().detach().numpy()

            batch_accuracy = (y_hat == y).mean()
            batch_accuracy_list.append(batch_accuracy)
            batch_loss_list.append(loss.item())

        model.eval()
        valid_batch_accuracy=[]
        valid_batch_loss = []
        with torch.no_grad():
            for n_valid, (X_valid, y_valid) in enumerate((valid_loader)):
                X_valid = torch.tensor(X_valid, device=device)#, dtype=torch.float32)
                y_valid = torch.tensor(y_valid, device=device, dtype=torch.float32)
                y_valid_hat = model(X_valid)
                
                valid_loss = criterion(y_valid_hat, y_valid).item()
                
                y_valid_hat = y_valid_hat.cpu().detach().numpy()>0.5
                                
                valid_batch_loss.append(valid_loss)
                valid_batch_accuracy.append((y_valid_hat == y_valid.cpu().detach().numpy()).mean())
                
            valid_losses.append(np.mean(valid_batch_loss))
            valid_accuracy.append(np.mean(valid_batch_accuracy))
            
        if np.mean(valid_batch_accuracy)>valid_best_accuracy:
            torch.save(model.state_dict(), 'C:/Study/dacon/computer2/data/csv/EfficientNetB7-fold{}.pt'.format(fold))
            valid_best_accuracy = np.mean(valid_batch_accuracy)
        print('fold : {}\tepoch : {:02d}\ttrain_accuracy / loss : {:.5f} / {:.5f}\tvalid_accuracy / loss : {:.5f} / {:.5f}\ttime : {:.0f}'.format(fold+1, epoch+1,
                                                                                                                                              np.mean(batch_accuracy_list),
                                                                                                                                              np.mean(batch_loss_list),
                                                                                                                                              np.mean(valid_batch_accuracy), 
                                                                                                                                              np.mean(valid_batch_loss),
                                                                                                                                              time.time()-start))

# ===================== Test Image 로드 ==========================
test_imgs=[]
for path in tqdm(test_imgs_dir):
    test_img=cv2.imread(path, cv2.IMREAD_COLOR)
    test_imgs.append(test_img)
test_imgs=np.array(test_imgs)

test_transform = transforms.Compose([
        transforms.ToTensor(),
        ])


# ================ Test 추론 =============================
submission = pd.read_csv('C:/Study/dacon/computer2/data/sample_submission.csv')

with torch.no_grad():
    for fold in range(1):
        model = EfficientNet_MultiLabel(in_channels=3).to(device)
        model.load_state_dict(torch.load('C:/Study/dacon/computer2/data/csv/EfficientNetB7-fold{}.pt'.format(fold)))
        model.eval()

        test_dataset = MnistDataset_v2(imgs = test_imgs, transform=test_transform, train=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

        for n, X_test in enumerate(tqdm(test_loader)):
            X_test = torch.tensor(X_test, device=device, dtype=torch.float32)
            with torch.no_grad():
                model.eval()
                pred_test = model(X_test).cpu().detach().numpy()
                submission.iloc[n*32:(n+1)*32,1:] += pred_test

# ==================== 제출물 생성 ====================
submission.iloc[:,1:] = np.where(submission.values[:,1:]>=0.5, 1,0)
submission.to_csv('C:/Study/dacon/computer2/data/csv/EfficientNetB7-fold0.csv', index=False)

# b7/ 1fold / 8batch / EfficientNetB7-fold0.csv
# 

''' 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator # 이미지데이터 늘리는 작업
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold
from keras import Sequential,Model
from keras.layers import *
from sklearn.decomposition import PCA
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, MaxPooling1D, Conv1D, AveragePooling2D
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('C:/computervision2/dirty_mnist_data/mnist_data/train.csv')
test = pd.read_csv('C:/computervision2/dirty_mnist_data/mnist_data/test.csv')
submission = pd.read_csv('C:/computervision2/dirty_mnist_data/mnist_data/submission.csv')

train2 = train.drop(['id','digit'],1) # 인덱스 있는 3개 버리기
test2 = test.drop(['id'],1) #인덱스 있는 것 버리기
train2 = train2.values
test2 = test2.values

train = np.concatenate([train2, test2], axis =0)
print(train.shape)

x = train[:,1:]
y = train[:,0]
x=np.array(x)
y1=np.array(y)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = OneHotEncoder()
y = encoder.fit_transform(y1.reshape(-1,1)).toarray()
print(y.shape)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66 )
print(type(x_test))
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)
x_train = x_train/255
x_train = x_train.astype('float32')
x_test = x_test/255
x_test = x_test.astype('float32')

idg = ImageDataGenerator(height_shift_range=(-1,1),width_shift_range=(-1,1)) # 이미지 카테고리화(4차원만 가능)
idg2 = ImageDataGenerator()

def modeling() :
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
    
    model.add(Flatten()) #2차원
    model.add(Dense(128,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(32,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(26,activation='softmax')) # softmax는 'categorical_crossentropy' 짝꿍
    return model
    
re = ReduceLROnPlateau(patience=10, verbose=1, factor= 0.5)
ea = EarlyStopping(patience=20, verbose=1, mode='auto')
epochs = 100
#KFlod대신 StratifiedKFold 써보기
#stratified 는 label 의 분포를 유지, 각 fold가 전체 데이터셋을 잘 대표한다.
skf = StratifiedKFold(n_splits=15, random_state=42, shuffle=True) #n_splits 몇 번 반복
val_loss_min = []
result = 0
nth = 0

train_generator = idg.flow(x_train[0:2000],y_train[0:2000],batch_size=1) #훈련데이터셋을 제공할 제네레이터를 지정
test_generator = idg2.flow(x_test,y_test) # validation_data에 넣을 것

model = modeling()
mc = ModelCheckpoint('../data/modelcheckpoint/0204_1_best_mc_4.h5', save_best_only=True, verbose=1)
model.compile(loss = 'categorical_crossentropy', optimizer=Adam(lr=0.002,epsilon=None) ,metrics=['acc']) # y의 acc가 목적
img_fit = model.fit(train_generator,batch_size=32,epochs=epochs,validation_data=test_generator, callbacks=[ea,mc,re])

# img_fit = model.fit_generator(train_generator,epochs=epochs, validation_data=test_generator, callbacks=[ea,mc,re])

# predict
model.load_weights('../data/modelcheckpoint/0204_1_best_mc_4.h5')
result += model.predict(test_generator,verbose=True) #a += b는 a= a+b
# predict_generator 예측 결과는 클래스별 확률 벡터로 출력
print('result:', result)
'''



