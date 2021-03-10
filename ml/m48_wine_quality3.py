# sklearn 할때는 y값 scaling 필요없음!!!!!!!!!!!!!!!!!!!!!!!!!!

import numpy as np
import pandas as pd

# csv불러오기
file_dir = 'C:/data/csv/wine'
wine = pd.read_csv(f'{file_dir}/winequality-white.csv', sep=';', index_col=None, header=0)
wine_test = pd.read_csv(f'{file_dir}/data-01-test-score.csv', sep=';', index_col=None, header=0)
# print(wine.shape)   #(4898, 12)
# print(wine.describe())


# pandas dataframe -> numpy 1
wine_npy = wine.to_numpy()


# numpy 슬라이싱
x = wine_npy[:,:-1]
y = wine_npy[:,-1]
print(x.shape, y.shape) #(4898, 11) (4898,)
print(np.unique(y)) #[3. 4. 5. 6. 7. 8. 9.]


# split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 66, shuffle = True)
print(x_train.shape, y_train.shape) #(3918, 11) (3918,)
print(x_test.shape, y_test.shape)   #(980, 11) (980,)

# scaling
from sklearn.preprocessing import QuantileTransformer, RobustScaler, MaxAbsScaler, PowerTransformer, StandardScaler
scale = QuantileTransformer()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)


# 모델구성
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# model = KNeighborsClassifier()   #score :  0.5663265306122449
model = RandomForestClassifier()   #score :  0.7142857142857143 ***
# model = XGBClassifier()          #score :  0.6816326530612244


model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('score : ', score)



