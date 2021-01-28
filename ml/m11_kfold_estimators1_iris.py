# m11_kfold_estimators1 복사
# all_estimators + Kfold, cross_val_score

# 모델 비교


from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_iris
import warnings

warnings.filterwarnings('ignore')

dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 66)

kfold = KFold(n_splits=5, shuffle=True)

# ====================================================all_estimators : sklearn에서 제공하는 분류형모델의 전체를 훈련
allAlgorithms = all_estimators(type_filter = 'classifier')

for (name, algorithm) in allAlgorithms:

    try:
        model = algorithm()

        scores = cross_val_score(model, x_train, y_train, cv=kfold)  
        print(name, '의 정답률: ', scores)

        # model.fit(x_train, y_train)
        # y_pred = model.predict(x_test)
        

    except:
        # continue
        print(name, '은 없는 놈!')

import sklearn
print(sklearn.__version__)      # 0.23.2

'''
AdaBoostClassifier 의 정답률 :  0.6333333333333333
BaggingClassifier 의 정답률 :  0.9333333333333333
BernoulliNB 의 정답률 :  0.3
CalibratedClassifierCV 의 정답률 :  0.9
CategoricalNB 의 정답률 :  0.9
CheckingClassifier 의 정답률 :  0.3

TypeError: __init__() missing 1 required positional argument: 'base_estimator'  

-->예외처리로 해결
'''



'''
Tensorflow                 :    이게 이겨야 돼
Dense, LSTM, Conv1D 모델 acc :  1.0

======================================================================================1. all_estimators

AdaBoostClassifier 의 정답률 :  0.6333333333333333
BaggingClassifier 의 정답률 :  0.9666666666666667
BernoulliNB 의 정답률 :  0.3
CalibratedClassifierCV 의 정답률 :  0.9
CategoricalNB 의 정답률 :  0.9
CheckingClassifier 의 정답률 :  0.3
ClassifierChain 은 없는 놈!
ComplementNB 의 정답률 :  0.6666666666666666
DecisionTreeClassifier 의 정답률 :  0.9333333333333333
DummyClassifier 의 정답률 :  0.3333333333333333
ExtraTreeClassifier 의 정답률 :  1.0
ExtraTreesClassifier 의 정답률 :  0.9333333333333333
GaussianNB 의 정답률 :  0.9666666666666667
GaussianProcessClassifier 의 정답률 :  0.9666666666666667
GradientBoostingClassifier 의 정답률 :  0.9666666666666667
HistGradientBoostingClassifier 의 정답률 :  0.8666666666666667
KNeighborsClassifier 의 정답률 :  0.9666666666666667
LabelPropagation 의 정답률 :  0.9333333333333333
LabelSpreading 의 정답률 :  0.9333333333333333
LinearDiscriminantAnalysis 의 정답률 :  1.0
LinearSVC 의 정답률 :  0.9666666666666667
LogisticRegression 의 정답률 :  1.0
LogisticRegressionCV 의 정답률 :  1.0
MLPClassifier 의 정답률 :  0.9666666666666667
MultiOutputClassifier 은 없는 놈!
MultinomialNB 의 정답률 :  0.9666666666666667
NearestCentroid 의 정답률 :  0.9333333333333333
NuSVC 의 정답률 :  0.9666666666666667
OneVsOneClassifier 은 없는 놈!
OneVsRestClassifier 은 없는 놈!
OutputCodeClassifier 은 없는 놈!
PassiveAggressiveClassifier 의 정답률 :  0.8
Perceptron 의 정답률 :  0.9333333333333333
QuadraticDiscriminantAnalysis 의 정답률 :  1.0
RadiusNeighborsClassifier 의 정답률 :  0.9666666666666667
RandomForestClassifier 의 정답률 :  0.9666666666666667
RidgeClassifier 의 정답률 :  0.8666666666666667
RidgeClassifierCV 의 정답률 :  0.8666666666666667
SGDClassifier 의 정답률 :  0.7
SVC 의 정답률 :  0.9666666666666667
StackingClassifier 은 없는 놈!
VotingClassifier 은 없는 놈!

======================================================================================2. all_estimators + Kfold
AdaBoostClassifier 의 정답률 :  [1.         0.95833333 0.83333333 0.95833333 0.95833333]
BaggingClassifier 의 정답률 :  [1.         0.95833333 0.79166667 1.         0.95833333]
BernoulliNB 의 정답률 :  [0.25       0.29166667 0.20833333 0.29166667 0.29166667]
CalibratedClassifierCV 의 정답률 :  [0.91666667 0.875      0.91666667 0.875      0.95833333]
CategoricalNB 의 정답률 :  [0.875      0.95833333 0.95833333 0.95833333 0.95833333]
CheckingClassifier 의 정답률 :  [0. 0. 0. 0. 0.]
ClassifierChain 은 없는 놈!
ComplementNB 의 정답률 :  [0.66666667 0.58333333 0.625      0.58333333 0.875     ]
DecisionTreeClassifier 의 정답률 :  [1.         0.95833333 0.95833333 0.875      0.95833333]
DummyClassifier 의 정답률 :  [0.20833333 0.41666667 0.45833333 0.125      0.29166667]
ExtraTreeClassifier 의 정답률 :  [1.         0.83333333 0.91666667 1.         1.        ]
ExtraTreesClassifier 의 정답률 :  [1.         1.         0.875      1.         0.91666667]
GaussianNB 의 정답률 :  [0.91666667 0.95833333 0.91666667 1.         0.95833333]
GaussianProcessClassifier 의 정답률 :  [1.         0.95833333 0.95833333 1.         0.91666667]
GradientBoostingClassifier 의 정답률 :  [0.91666667 1.         0.91666667 0.95833333 0.875     ]
HistGradientBoostingClassifier 의 정답률 :  [0.95833333 0.95833333 0.875      0.91666667 0.875     ]
KNeighborsClassifier 의 정답률 :  [1.         1.         0.95833333 0.95833333 0.875     ]
LabelPropagation 의 정답률 :  [0.91666667 0.91666667 0.875      1.         1.        ]
LabelSpreading 의 정답률 :  [0.91666667 0.95833333 0.95833333 1.         1.        ]
LinearDiscriminantAnalysis 의 정답률 :  [1.         1.         0.95833333 0.91666667 1.        ]
LinearSVC 의 정답률 :  [0.95833333 0.95833333 0.95833333 1.         0.95833333]
LogisticRegression 의 정답률 :  [0.95833333 0.83333333 1.         0.95833333 0.95833333]
LogisticRegressionCV 의 정답률 :  [1.         0.95833333 1.         0.875      1.        ]
MLPClassifier 의 정답률 :  [0.95833333 1.         1.         0.95833333 0.91666667]
MultiOutputClassifier 은 없는 놈!
MultinomialNB 의 정답률 :  [0.66666667 0.83333333 0.79166667 0.58333333 0.91666667]
NearestCentroid 의 정답률 :  [1.         0.91666667 0.875      0.875      0.91666667]
NuSVC 의 정답률 :  [1.         0.95833333 0.95833333 0.91666667 0.95833333]
OneVsOneClassifier 은 없는 놈!
OneVsRestClassifier 은 없는 놈!
OutputCodeClassifier 은 없는 놈!
PassiveAggressiveClassifier 의 정답률 :  [0.95833333 0.58333333 0.625      1.         0.91666667]
Perceptron 의 정답률 :  [0.79166667 0.875      0.625      0.45833333 0.66666667]
QuadraticDiscriminantAnalysis 의 정답률 :  [0.95833333 0.95833333 1.         0.95833333 0.95833333]
RadiusNeighborsClassifier 의 정답률 :  [0.95833333 0.95833333 1.         0.95833333 1.        ]
RandomForestClassifier 의 정답률 :  [1.         0.95833333 0.95833333 0.91666667 0.91666667]
RidgeClassifier 의 정답률 :  [0.75       0.75       0.91666667 0.66666667 0.95833333]
RidgeClassifierCV 의 정답률 :  [0.75       0.79166667 0.83333333 0.75       0.875     ]
SGDClassifier 의 정답률 :  [0.58333333 0.70833333 0.95833333 0.83333333 0.33333333]
SVC 의 정답률 :  [0.95833333 0.91666667 0.95833333 1.         1.        ]
StackingClassifier 은 없는 놈!
VotingClassifier 은 없는 놈!

'''