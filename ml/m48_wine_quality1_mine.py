# 실습
# 맹그러봐
# -------------------내가 test 한 코드
# Y 7가지
# Y값이 적으면 정확도가 오름
# to_categorical -> OneHotEncoder 사용

import numpy as np
import pandas as pd

# csv불러오기
file_dir = 'C:/data/csv/wine'
df = pd.read_csv(f'{file_dir}/winequality-white.csv', sep=';')
df_test = pd.read_csv(f'{file_dir}/data-01-test-score.csv', sep=';')
# print(df)          #(4898, 12)
# print(df.info())   # x: float64, y: int64


# pandas dataframe -> numpy 1
aaa = df.to_numpy()
# print(aaa)        # target 값 float로 바뀜 (numpy 한가지 형태)
# print(aaa.shape)  # (4898, 12)
# print(type(aaa))  # <class 'numpy.ndarray'>
# np.save('../data/npy/iris_sklearn.npy', arr=aaa)


# numpy 슬라이싱
x = aaa[:,:-1]
y = aaa[:,-1]
# print(x)
# print(y)  

print(np.unique(y)) #[3. 4. 5. 6. 7. 8. 9.]

# 사사분위활용 이상치 찾기----> 많다
''' def outliers (data_out):
    outlier_loc_list = []
    for col in range(data_out.shape[1]):
        print(data_out[:,col])
        quartile_1, quartile_2, quartile_3 = np.percentile(data_out[:,col], [25,50,75])    #percentile:지정된 축을 따라 데이터의 q 번째 백분위 수를 계산합니다.
        print('1사분위 : ', quartile_1)
        print('2사분위 : ', quartile_2)
        print('3사분위 : ', quartile_3)

        iqr = quartile_3 - quartile_1   # 3사분위 - 1사분위
        
        # 양방향으로 1.5배씩 늘려서 정상적인 데이터범위 지정
        # 통상 1.5
        lower_bound = quartile_1 - (iqr * 1.5)  
        upper_bound = quartile_3 + (iqr * 1.5)

        outlier_loc = np.where((data_out[:,col]>upper_bound) | (data_out[:,col]<lower_bound))
        print(outlier_loc)
        outlier_loc_list.append(outlier_loc)
    return np.array(outlier_loc_list)


outlier_loc_list = outliers(aaa)
print('이상치의 위치 : ', outlier_loc_list)

import matplotlib.pyplot as plt
plt.boxplot(x)
plt.show() '''


# 전처리

#PCA : 컬럼 재구성(압축)
from sklearn.decomposition import PCA
pca = PCA(n_components=6)
x = pca.fit_transform(x)

# y (-1, 7)로 맞추자!
# y 전처리
from sklearn.preprocessing import OneHotEncoder
y = y.reshape(-1,1)
onehot = OneHotEncoder()
onehot.fit(y)
y = onehot.transform(y).toarray()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 120, shuffle = True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.6, random_state = 120, shuffle = True)
# print(x_train.shape, y_train.shape) #(2350, 11) (2350,)
# print(x_test.shape, y_test.shape)   #(980, 11) (980,)
# print(x_val.shape, y_val.shape)     #(1568, 11) (1568,)


#x전처리
from sklearn.preprocessing import QuantileTransformer, RobustScaler, MaxAbsScaler, PowerTransformer
# scaler = PowerTransformer(method='yeo-johnson')
scaler = QuantileTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

import matplotlib.pyplot as plt
plt.boxplot(x)
# plt.show()


# parameters       
parameters = [ 
    {'n_estimators':[100,200,300], 'min_samples_split':[2,3,4,5], 'n_jobs':[-1]},  
    {'n_estimators':[1,100],    'max_depth':[35,40,44], 'min_samples_leaf':[2,4,5], 'min_samples_split':[8,10], 'n_jobs':[-1]},
    {'n_estimators':[100,200], 'min_samples_leaf':[12,24]},

]

#-----------------------------------------DecisionTreeClassifier
from sklearn.utils.testing import all_estimators
from sklearn.metrics import accuracy_score
allAlgorithms = all_estimators(type_filter = 'classifier')

for (name, algorithm) in allAlgorithms:

    try:
        model = algorithm()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률      : ', accuracy_score(y_test, y_pred))

    except:
        print(name, '은 없는 놈!') 



#2.모델구성
#-----------------------------------------ML
''' from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

# model = GridSearchCV(DecisionTreeClassifier(), parameters, verbose=1)
model = DecisionTreeClassifier()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가
results = model.score(x_test, y_test)

print('score       : ', results)
# score       :  0.5469387755102041 '''

