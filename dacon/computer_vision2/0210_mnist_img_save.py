import os
import cv2
import pandas as pd

# 1. 데이터
csv_train = pd.read_csv('./dacon/computer2/data/mnist_data/train.csv', header=0)
csv_test = pd.read_csv('./dacon/computer2/data/mnist_data/test.csv', header=0)
submit = pd.read_csv('./dacon/computer2/data/mnist_data/submission.csv', header=0)

path_img = './dacon/computer2/data/img'

os.mkdir(f'{path_img}/images_train')
os.mkdir(f'{path_img}/images_train/0')
os.mkdir(f'{path_img}/images_train/1')
os.mkdir(f'{path_img}/images_train/2')
os.mkdir(f'{path_img}/images_train/3')
os.mkdir(f'{path_img}/images_train/4')
os.mkdir(f'{path_img}/images_train/5')
os.mkdir(f'{path_img}/images_train/6')
os.mkdir(f'{path_img}/images_train/7')
os.mkdir(f'{path_img}/images_train/8')
os.mkdir(f'{path_img}/images_train/9')
os.mkdir(f'{path_img}/images_test')

for idx in range(len(csv_train)) :
    img = csv_train.loc[idx, '0':].values.reshape(28, 28).astype(int)
    digit = csv_train.loc[idx, 'digit']
    cv2.imwrite(f'{path_img}/images_train/{digit}/{csv_train["id"][idx]}.png', img)


for idx in range(len(csv_test)) :
    img = csv_test.loc[idx, '0':].values.reshape(28, 28).astype(int)
    cv2.imwrite(f'{path_img}/images_test/{csv_test["id"][idx]}.png', img)