# mnist1 train데이터로 이미지저장하고
# 증폭시킨다음 훈련시키고
# mnist2데이터 예측값으로 넣어보자
# 다중라벨분류?

import os
import cv2
import pandas as pd

# 1. 데이터
csv_train = pd.read_csv('C:/Study/dacon/computer2/data/mnist_data/train.csv', header=0)
csv_test = pd.read_csv('C:/Study/dacon/computer2/data/mnist_data/test.csv', header=0)
submit = pd.read_csv('C:/Study/dacon/computer2/data/mnist_data/submission.csv', header=0)

path_img = 'C:/Study/dacon/computer2/data/mnist_data/img'

os.mkdir(f'{path_img}/minist1_train')
os.mkdir(f'{path_img}/minist1_train/A')
os.mkdir(f'{path_img}/minist1_train/B')
os.mkdir(f'{path_img}/minist1_train/C')
os.mkdir(f'{path_img}/minist1_train/D')
os.mkdir(f'{path_img}/minist1_train/E')
os.mkdir(f'{path_img}/minist1_train/F')
os.mkdir(f'{path_img}/minist1_train/G')
os.mkdir(f'{path_img}/minist1_train/H')
os.mkdir(f'{path_img}/minist1_train/I')
os.mkdir(f'{path_img}/minist1_train/J')
os.mkdir(f'{path_img}/minist1_train/K')
os.mkdir(f'{path_img}/minist1_train/L')
os.mkdir(f'{path_img}/minist1_train/M')
os.mkdir(f'{path_img}/minist1_train/N')
os.mkdir(f'{path_img}/minist1_train/O')
os.mkdir(f'{path_img}/minist1_train/P')
os.mkdir(f'{path_img}/minist1_train/Q')
os.mkdir(f'{path_img}/minist1_train/R')
os.mkdir(f'{path_img}/minist1_train/S')
os.mkdir(f'{path_img}/minist1_train/T')
os.mkdir(f'{path_img}/minist1_train/U')
os.mkdir(f'{path_img}/minist1_train/V')
os.mkdir(f'{path_img}/minist1_train/W')
os.mkdir(f'{path_img}/minist1_train/X')
os.mkdir(f'{path_img}/minist1_train/Y')
os.mkdir(f'{path_img}/minist1_train/Z')

for idx in range(len(csv_train)) :
    img = csv_train.loc[idx, '0':].values.reshape(28, 28).astype(int)
    letter = csv_train.loc[idx, 'letter']
    cv2.imwrite(f'{path_img}/minist1_train/{letter}/{csv_train["id"][idx]}.png', img)


# for idx in range(len(csv_test)) :
#     img = csv_test.loc[idx, '0':].values.reshape(28, 28).astype(int)
#     cv2.imwrite(f'{path_img}/images_test/{csv_test["id"][idx]}.png', img)