# m11_kfold_estimators1 복사
# all_estimators + Kfold, cross_val_score

# 모델 비교


from sklearn.datasets import load_wine
from sklearn.utils.testing import all_estimators
from sklearn. model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

dataset = load_wine()

x = dataset.data
y = dataset.target 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 66)

kfold = KFold(n_splits=5, shuffle=True)

allAlgorithms = all_estimators(type_filter = 'classifier')

for (name, algorithm) in allAlgorithms:

    try:
        model = algorithm()

        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        print(name, '의 정답률      : ', scores)

        # model.fit(x_train, y_train)
        # y_pred = model.predict(x_test)

    except:
        print(name, '은 없는 놈!')


'''
Tensorflow                 :
Dense모델 acc :  1.0

======================================================================================1. all_estimators

AdaBoostClassifier 의 정답률      :  0.8888888888888888
BaggingClassifier 의 정답률      :  1.0
BernoulliNB 의 정답률      :  0.4166666666666667
CalibratedClassifierCV 의 정답률      :  0.9444444444444444
CategoricalNB 은 없는 놈!
CheckingClassifier 의 정답률      :  0.3888888888888889
ClassifierChain 은 없는 놈!
ComplementNB 의 정답률      :  0.6944444444444444
DecisionTreeClassifier 의 정답률      :  0.9444444444444444
DummyClassifier 의 정답률      :  0.2777777777777778
ExtraTreeClassifier 의 정답률      :  0.8611111111111112
ExtraTreesClassifier 의 정답률      :  1.0
GaussianNB 의 정답률      :  1.0
GaussianProcessClassifier 의 정답률      :  0.4444444444444444
GradientBoostingClassifier 의 정답률      :  0.9722222222222222
HistGradientBoostingClassifier 의 정답률      :  0.9722222222222222
KNeighborsClassifier 의 정답률      :  0.6944444444444444
LabelPropagation 의 정답률      :  0.5277777777777778
LabelSpreading 의 정답률      :  0.5277777777777778
LinearDiscriminantAnalysis 의 정답률      :  1.0
LinearSVC 의 정답률      :  0.6388888888888888
LogisticRegression 의 정답률      :  0.9722222222222222
LogisticRegressionCV 의 정답률      :  0.9722222222222222
MLPClassifier 의 정답률      :  0.4166666666666667
MultiOutputClassifier 은 없는 놈!
MultinomialNB 의 정답률      :  0.7777777777777778
NearestCentroid 의 정답률      :  0.6944444444444444
NuSVC 의 정답률      :  0.9444444444444444
OneVsOneClassifier 은 없는 놈!
OneVsRestClassifier 은 없는 놈!
OutputCodeClassifier 은 없는 놈!
PassiveAggressiveClassifier 의 정답률      :  0.6944444444444444
Perceptron 의 정답률      :  0.6388888888888888
QuadraticDiscriminantAnalysis 의 정답률      :  0.9722222222222222
RadiusNeighborsClassifier 은 없는 놈!
RandomForestClassifier 의 정답률      :  1.0
RidgeClassifier 의 정답률      :  1.0
RidgeClassifierCV 의 정답률      :  1.0
SGDClassifier 의 정답률      :  0.4722222222222222
SVC 의 정답률      :  0.6944444444444444
StackingClassifier 은 없는 놈!
VotingClassifier 은 없는 놈!

======================================================================================2. all_estimators + Kfold
AdaBoostClassifier 의 정답률      :  [0.82758621 0.96551724 0.85714286 0.57142857 0.89285714]
BaggingClassifier 의 정답률      :  [0.89655172 0.86206897 0.96428571 0.85714286 0.85714286]
BernoulliNB 의 정답률      :  [0.4137931  0.44827586 0.35714286 0.35714286 0.39285714]
CalibratedClassifierCV 의 정답률      :  [1.         0.89655172 0.96428571 0.89285714 0.85714286]
CategoricalNB 은 없는 놈!
CheckingClassifier 의 정답률      :  [0. 0. 0. 0. 0.]
ClassifierChain 은 없는 놈!
ComplementNB 의 정답률      :  [0.44827586 0.68965517 0.82142857 0.78571429 0.57142857]
DecisionTreeClassifier 의 정답률      :  [0.86206897 0.89655172 0.92857143 0.89285714 0.89285714]
DummyClassifier 의 정답률      :  [0.37931034 0.20689655 0.21428571 0.32142857 0.25      ]
ExtraTreeClassifier 의 정답률      :  [0.82758621 0.82758621 0.92857143 0.92857143 0.89285714]
ExtraTreesClassifier 의 정답률      :  [1.         1.         0.96428571 1.         0.85714286]
GaussianNB 의 정답률      :  [0.96551724 0.93103448 1.         0.96428571 1.        ]
GaussianProcessClassifier 의 정답률      :  [0.4137931  0.37931034 0.32142857 0.46428571 0.53571429]
GradientBoostingClassifier 의 정답률      :  [0.79310345 0.82758621 1.         0.96428571 0.92857143]
HistGradientBoostingClassifier 의 정답률      :  [0.96551724 0.96551724 1.         0.96428571 1.        ]
KNeighborsClassifier 의 정답률      :  [0.75862069 0.62068966 0.60714286 0.67857143 0.82142857]
LabelPropagation 의 정답률      :  [0.48275862 0.44827586 0.42857143 0.42857143 0.42857143]
LabelSpreading 의 정답률      :  [0.48275862 0.51724138 0.35714286 0.5        0.28571429]
LinearDiscriminantAnalysis 의 정답률      :  [1.         0.96551724 0.96428571 1.         0.92857143]
LinearSVC 의 정답률      :  [0.79310345 0.86206897 0.85714286 0.42857143 0.57142857]
LogisticRegression 의 정답률      :  [0.93103448 0.89655172 0.92857143 0.92857143 0.96428571]
LogisticRegressionCV 의 정답률      :  [0.89655172 0.89655172 1.         0.92857143 0.92857143]
MLPClassifier 의 정답률      :  [0.82758621 0.27586207 0.32142857 0.92857143 1.        ]
MultiOutputClassifier 은 없는 놈!
MultinomialNB 의 정답률      :  [0.86206897 0.82758621 0.92857143 0.85714286 0.89285714]
NearestCentroid 의 정답률      :  [0.65517241 0.75862069 0.75       0.71428571 0.75      ]
NuSVC 의 정답률      :  [0.89655172 0.82758621 0.92857143 0.85714286 0.78571429]
OneVsOneClassifier 은 없는 놈!
OneVsRestClassifier 은 없는 놈!
OutputCodeClassifier 은 없는 놈!
PassiveAggressiveClassifier 의 정답률      :  [0.55172414 0.34482759 0.71428571 0.28571429 0.42857143]
Perceptron 의 정답률      :  [0.4137931  0.68965517 0.42857143 0.53571429 0.82142857]
QuadraticDiscriminantAnalysis 의 정답률      :  [0.96551724 0.93103448 1.         1.         1.        ]
RadiusNeighborsClassifier 은 없는 놈!
RandomForestClassifier 의 정답률      :  [0.96551724 0.96551724 0.96428571 0.96428571 0.96428571]
RidgeClassifier 의 정답률      :  [1.         1.         1.         0.96428571 0.96428571]
RidgeClassifierCV 의 정답률      :  [1.         0.96551724 1.         1.         0.96428571]
SGDClassifier 의 정답률      :  [0.68965517 0.72413793 0.42857143 0.46428571 0.64285714]
SVC 의 정답률      :  [0.75862069 0.5862069  0.5        0.67857143 0.64285714]
StackingClassifier 은 없는 놈!
VotingClassifier 은 없는 놈!


'''