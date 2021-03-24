# https://www.kaggle.com/ignatovdv/cnn-pytorch-simpsons?select=simspsons.csv

# ==============================================최적의 LR 범위를 찾는 기능
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms
from torchvision import datasets, models, transforms

import PIL
from PIL import Image

import math
import random
import seaborn as sn
import pandas as pd
import numpy as np
from pathlib import Path
from skimage import io
import pickle
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm_notebook
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Ignore warnings
import warnings


# 샘플을 train, val 및 test로 나눕니다.
class SimpsonTrainValPath():
    
    def __init__(self, train_dir, test_dir):

        self.train_dir = train_dir
        self.test_dir = test_dir
        self.train_val_files_path = sorted(list(self.train_dir.rglob('*.jpg')))
        self.test_path = sorted(list(self.test_dir.rglob('*.jpg')))
        self.train_val_labels = [path.parent.name for path in self.train_val_files_path]

    def get_path(self):
            
        train_files_path, val_files_path = train_test_split(self.train_val_files_path, test_size = 0.3, \
                                                stratify=self.train_val_labels)

        files_path = {'train': train_files_path, 'val': val_files_path}

        return files_path, self.test_path

    def get_n_classes(self):
        return len(np.unique(self.train_val_labels))


# 데이터 생성
class SimpsonsDataset(Dataset):
    
    def __init__(self, files_path, data_transforms):
        self.files_path = files_path
        self.transform = data_transforms
        
        if 'test' not in str(self.files_path[0]):
            self.labels = [path.parent.name for path in self.files_path]
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.labels)
            
        with open('C:/data/kaggle/input/the-simpsons-characters-dataset/label_encoder.pkl', 'wb') as le_dump_file:
            pickle.dump(self.label_encoder, le_dump_file)

    def __len__(self):
        return len(self.files_path)

    def __getitem__(self, idx):
        img_path = str(self.files_path[idx]) 
        image = Image.open(img_path)
        image = self.transform(image)
        
        if 'test' in str(self.files_path[0]):
            return image
        else: 
            label_str = str(self.files_path[idx].parent.name)
            label = self.label_encoder.transform([label_str]).item()
        
        return image, label



if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)
    print("Pillow Version: ", PIL.PILLOW_VERSION)
    # PyTorch Version:  1.7.1+cu101
    # Torchvision Version:  0.8.2+cu101
    # Pillow Version:  8.0.1

    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')
        # CUDA is available!  Training on GPU ...

    # 데이터 경로 형성
    # input / ... 데이터 경로는 게시 된 데이터 세트 kaggle.com/alexattia/the-simpsons-characters-dataset의 링크와 동일합니다.
    train_dir = Path('C:/data/kaggle/input/the-simpsons-characters-dataset/simpsons_dataset/')
    test_dir = Path('C:/data/kaggle/input/the-simpsons-characters-dataset/kaggle_simpson_testset/')

    SimpsonTrainValPath = SimpsonTrainValPath(train_dir, test_dir)
    train_path, test_path = SimpsonTrainValPath.get_path()  


    # 학습 모듈
    def train_model(model, dataloaders, criterion, optimizer, save_best_weights_path, save_last_weights_path, best_acc, num_epochs=25, is_inception=False):
        since = time.time()

        val_acc_history = []
        val_loss_history = []
        train_acc_history = []
        train_loss_history = []
        lr_find_lr = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # 모델 모드 설정
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # train모드
                else:
                    model.eval()   # 평가모드

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in tqdm_notebook(dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # 그래디언트 값 업데이트
                    optimizer.zero_grad()

                    # forward
                    # 훈련 모드 인 경우 기록 저장
                    with torch.set_grad_enabled(phase == 'train'):
                        # 손실 계산
                        # 모델 개시 사례
                        if is_inception and phase == 'train':
                            outputs, aux_outputs = model(inputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4*loss2
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # 역방향 + 최적화 + 스케줄러 (훈련 모드 인 경우)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                            lr_step = optimizer_ft.state_dict()["param_groups"][0]["lr"]
                            lr_find_lr.append(lr_step)

                    # stats
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                # loss, acc 
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # 최고의 모델 유지
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                # 유지 acc 
                if phase == 'val':
                    val_acc_history.append(epoch_acc)
                    val_loss_history.append(epoch_loss)
                else:
                    train_acc_history.append(epoch_acc)
                    train_loss_history.append(epoch_loss)
            
            print()
        # 훈련 시간 출력
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # 최상의 가중치로 모델로드
        model.load_state_dict(best_model_wts)

        history_val = {'loss': val_loss_history, 'acc': val_acc_history}
        history_train = {'loss': train_loss_history, 'acc': train_acc_history}
        
        return model, history_val, history_train, time_elapsed, lr_find_lr, best_acc


    # 기울기 계산 비활성화
    # feature_extracting = False - 전체 네트워크의 기울기 계산
    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False


    # 모델
    def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
        
        model_ft = None
        input_size = 0
        
        if model_name == "resnet152":
            """ Resnet152
            """
            model_ft = models.resnet152(pretrained=use_pretrained)
            # 고정 된 레이어에 대한 그라디언트 업데이트 비활성화
            set_parameter_requires_grad(model_ft, feature_extract)
            # получаем кол-во нейронов входящих в последний слой
            num_ftrs = model_ft.fc.in_features
            # 마지막 레이어에서 필요한 출력을 설정하십시오.
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224
        
        elif model_name == 'resnext-101-32x8d':
            """ ResNeXt-101-32x8d
            """
            model_ft = models.resnext101_32x8d(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "vgg":
            """ VGG11_bn
            """
            model_ft = models.vgg11_bn(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "squeezenet":
            """ Squeezenet
            """
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
            model_ft.num_classes = num_classes
            input_size = 224

        elif model_name == "densenet":
            """ Densenet
            """
            model_ft = models.densenet161(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft, input_size


    # 모델 초기화 및 데이터 준비
    # 모델 설정
    # 같은 (model_name) 모델 이름으로 Google 드라이브에 폴더를 만들어 저장합니다.

    # initialize_model에 정의 된 모델 중에서 선택
    model_name = 'resnet152'

    # 모델 저장에 대한 추가 설명
    fc_layer = 'all-st-SGD-m.9-nest-s-cycle-exp-.00001-.05-g.99994-m.8-.9'

    # 데이터 세트의 클래스 수
    num_classes = SimpsonTrainValPath.get_n_classes()

    # batch_size, epochs
    batch_size = 32
    num_epochs = 2

    # 설정 장치
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # feature_extract = False - 전체 모델 학습
    # feature_extract = True - 훈련 FC
    feature_extract = False

    # 모델저장
    save_last_weights_path = 'C:/data/kaggle/working/' + model_name + '-' + fc_layer + '_last_weights.pth'
    save_best_weights_path = 'C:/data/kaggle/working/' + model_name + '-' + fc_layer + '_best_weights.pth'

    # 모델 초기화
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    # 모델을 GPU로 전송
    model_ft = model_ft.to(device)


    # 데이터 증대
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.RandomChoice( [ 
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ColorJitter(contrast=0.9),
                                    transforms.ColorJitter(brightness=0.1),
                                    transforms.RandomApply( [ transforms.RandomHorizontalFlip(p=1), transforms.ColorJitter(contrast=0.9) ], p=0.5),
                                    transforms.RandomApply( [ transforms.RandomHorizontalFlip(p=1), transforms.ColorJitter(brightness=0.1) ], p=0.5),
                                    ] ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


    # 네트워크에 제출할 데이터

    # Dataset
    image_datasets = {mode: SimpsonsDataset(train_path[mode], data_transforms[mode]) for mode in ['train', 'val']}
    image_datasets_test = SimpsonsDataset(test_path, data_transforms['val'])


    # Dataloader
    dataloaders_dict = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
                        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=True, num_workers=4)}
    dataloader_test = torch.utils.data.DataLoader(image_datasets_test, batch_size=batch_size, shuffle=False, num_workers=4)


    # 데이터 시각화
    def imshow(inp, title=None, plt_ax=plt, default=False):
        """텐서 용 Imshow"""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt_ax.imshow(inp)
        if title is not None:
            plt_ax.set_title(title)
        plt_ax.grid(False)
    fig, ax = plt.subplots(nrows=3, ncols=3,figsize=(8, 8), \
                            sharey=True, sharex=True)
    for fig_x in ax.flatten():
        random_characters = int(np.random.uniform(0, 4500))
        im_val, label = image_datasets['train'][random_characters]
        # inverse_transform은 LabelEncoder () 메소드이며, inverse_transform을 사용하여 클래스를 숫자로 인코딩하고 숫자에서 클래스 이름을 반환합니다.
        # 대문자에서 문자 이름 가져 오기
        img_label = " ".join(map(lambda x: x.capitalize(),\
                    image_datasets['val'].label_encoder.inverse_transform([label])[0].split('_')))
        imshow(im_val.data.cpu(), \
            title=img_label,plt_ax=fig_x)


    # 학습 시각화
    def visualization(train, val, is_loss = True):
        if is_loss:
            plt.figure(figsize=(17,10))
            plt.plot(train, label = 'Training loss')
            plt.plot(val, label = 'Val loss')
            plt.title('Training and validation loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
        
        else:
            plt.figure(figsize=(17,10))
            plt.plot(train, label = 'Training acc')
            plt.plot(val, label = 'Val acc')
            plt.title('Training and validation acc')
            plt.xlabel('Epochs')
            plt.ylabel('Acc')
            plt.legend()
            plt.show()

    # 최적화 매개 변수
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                #print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                pass
                #print("\t",name)


    # 연습
    # 운동은 두 단계로 구성됩니다.

    # Pytorch 라이브러리에서 lr_scheduler.CyclicLR을 사용하여 주기적으로 변경되는 lr의 최적 범위 찾기
    # 찾은 범위를 사용하여 모델 훈련
    # 최적의 학습률 (lr) 결정
    # 지정된 범위의 값을 주기적으로 변경하여 lr을 최적화합니다. 먼저 최적의 범위 경계를 결정해야합니다. 이를 위해 테스트 실행을 수행합니다. 
    # 테스트 실행 중에 lr이 선형 적으로 증가하도록 단계 크기를 선택해 보겠습니다.

    # 그런 다음 그래프를 기반으로 정확도 대 학습률 그래프를 작성하고 값 범위를 선택합니다.
    #  https://arxiv.org/pdf/1506.01186.pdf

    base_lr = 0.00001
    max_lr = 0.05
    lr_find_epochs = 2

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9, nesterov = True)

    step_size = lr_find_epochs * len(dataloaders_dict['train'])

    scheduler = optim.lr_scheduler.CyclicLR(optimizer_ft, base_lr = base_lr, max_lr = max_lr, step_size_up=step_size, 
                                        mode='exp_range', gamma=0.99994, scale_mode='cycle', cycle_momentum=True, 
                                        base_momentum=0.8, max_momentum=0.9, last_epoch=-1)


    # 최적의 LR 범위를 찾는 기능
    def search_lr(lr_find_epochs):
        
        accs = []
        lr_find_lr = []
        acc_sum = 0.0

        for i in range(lr_find_epochs):
            print("epoch {}".format(i))
            for inputs, labels in tqdm_notebook(dataloaders_dict['train']):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                model_ft.train()
                optimizer_ft.zero_grad()
                
                outputs = model_ft(inputs)
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, 1)
                acc_running = torch.sum(preds == labels.data).item()
                acc_sum += torch.sum(preds == labels.data).item()

                loss.backward()
                optimizer_ft.step()
                scheduler.step()
                
                lr_step = optimizer_ft.state_dict()["param_groups"][0]["lr"]
                lr_find_lr.append(lr_step)
                
                accs.append(acc_running)
        accs = np.array(accs) / acc_sum
        return lr_find_lr, accs

    lr_find_lr, accs = search_lr(lr_find_epochs)

    plt.figure(figsize=(20,10))
    plt.plot(np.array(lr_find_lr), np.array(accs))

    # 정확도가 증가하는 lr 범위를 살펴 보겠습니다. 


