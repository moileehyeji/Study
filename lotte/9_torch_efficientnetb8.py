# https://dacon.io/competitions/official/235697/codeshare/2429?page=2&dtype=recent

import os
import easydict
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import random
import tqdm
import cv2
from efficientnet_pytorch import EfficientNet
import torch
import albumentations
import albumentations.pytorch


x = np.load("C:/Study/lotte/data/npy/128_project_x.npy",allow_pickle=True)
x_pred = np.load('C:/Study/lotte/data/npy/128_test.npy',allow_pickle=True)
y = np.load("C:/Study/lotte/data/npy/128_project_y.npy",allow_pickle=True)

x = x.reshape(-1, 128*128)

x_df = pd.DataFrame(data=x)
y_df = pd.DataFrame(data=y)
#=============================================
x_df.to_csv('C:/Study/lotte/data/9_x.csv', index=True, encoding='cp949')
y_df.to_csv('C:/Study/lotte/data/9_y.csv', index=True, encoding='cp949')
#=============================================

print(x)
print(y)