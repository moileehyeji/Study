import numpy as np
import pandas as pd

# csv불러오기
file_dir = 'C:/data/csv/wine'
wine = pd.read_csv(f'{file_dir}/winequality-white.csv', sep=';', index_col=None, header=0)
wine_test = pd.read_csv(f'{file_dir}/data-01-test-score.csv', sep=';', index_col=None, header=0)
# print(wine.shape)   #(4898, 12)
# print(wine.describe())

''' # pandas.groupby 
# 객체를 분할하는 기능을 적용하고, 그 결과 결합의 조합을 포함한다. 
# 이는 대량의 데이터를 그룹화하고 이러한 그룹에 대한 연산을 계산하는 데 사용할 수 있습니다.
count_data = wine.groupby('quality')['quality'].count()
print(count_data)
# 3      20
# 4     163
# 5    1457
# 6    2198
# 7     880
# 8     175
# 9       5
import matplotlib.pyplot as plt 
count_data.plot()
plt.show()
# category 즉, 데이터를 조절할 수 있는 권한이 있을때만 가능 '''


# pandas dataframe -> numpy 1
wine_npy = wine.to_numpy()


# numpy 슬라이싱
# x = wine_npy[:,:-1]
# y = wine_npy[:,-1]
x = wine.drop('quality', axis = 1)
y = wine['quality']
print(x.shape, y.shape) #(4898, 11) (4898,)
print(np.unique(y)) #[3. 4. 5. 6. 7. 8. 9.]


# ----------------------------------------------y category 줄여서 라벨링
# list(y) -> [3. 4. 5. 6. 7. 8. 9.]
newlist = []
for i in list(y):
    if   i <= 4:  newlist += [0]
    elif i <= 7:  newlist += [1]
    elif i <= 9:  newlist += [2]
y = np.array(newlist)
print(x.shape, y.shape) #(4898, 11) (4898,)
# ----------------------------------------------

 
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
model = RandomForestClassifier()   #score :  0.7142857142857143 ->  0.9530612244897959
# model = XGBClassifier()          #score :  0.6816326530612244


model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('score : ', score)




# y 카테고리 줄이기 전  score :  0.7142857142857143 ***
# y 카테고리 줄이기 후  score :  0.9530612244897959




 