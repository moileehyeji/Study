# m11_kfold_estimators1 복사
# all_estimators + Kfold, cross_val_score

# 모델 비교

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.utils.testing import all_estimators
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

dataset = load_breast_cancer()
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
        print(name, '의 정답률 : ', scores)

        # model.fit(x_train, y_train)
        # y_pred = model.predict(x_test)
        

    except:
        print(name, '은 없는 놈!')

'''
Tensorflow                 :
Dense(Dropout)모델 acc :  0.9912280440330505

======================================================================================1. all_estimators

AdaBoostClassifier 의 정답률 :  0.9473684210526315
BaggingClassifier 의 정답률 :  0.9298245614035088
BernoulliNB 의 정답률 :  0.6403508771929824
CalibratedClassifierCV 의 정답률 :  0.8859649122807017
CategoricalNB 은 없는 놈!
CheckingClassifier 의 정답률 :  0.35964912280701755
ClassifierChain 은 없는 놈!
ComplementNB 의 정답률 :  0.868421052631579
DecisionTreeClassifier 의 정답률 :  0.9122807017543859
DummyClassifier 의 정답률 :  0.5614035087719298
ExtraTreeClassifier 의 정답률 :  0.9035087719298246
ExtraTreesClassifier 의 정답률 :  0.9649122807017544
GaussianNB 의 정답률 :  0.9385964912280702
GaussianProcessClassifier 의 정답률 :  0.8771929824561403
GradientBoostingClassifier 의 정답률 :  0.9473684210526315
HistGradientBoostingClassifier 의 정답률 :  0.9736842105263158      ******
KNeighborsClassifier 의 정답률 :  0.9210526315789473
LabelPropagation 의 정답률 :  0.3684210526315789
LabelSpreading 의 정답률 :  0.3684210526315789
LinearDiscriminantAnalysis 의 정답률 :  0.9473684210526315
LinearSVC 의 정답률 :  0.8947368421052632
LogisticRegression 의 정답률 :  0.9385964912280702
LogisticRegressionCV 의 정답률 :  0.956140350877193
MLPClassifier 의 정답률 :  0.9122807017543859
MultiOutputClassifier 은 없는 놈!
MultinomialNB 의 정답률 :  0.8596491228070176
NearestCentroid 의 정답률 :  0.868421052631579
NuSVC 의 정답률 :  0.8596491228070176
OneVsOneClassifier 은 없는 놈!
OneVsRestClassifier 은 없는 놈!
OutputCodeClassifier 은 없는 놈!
PassiveAggressiveClassifier 의 정답률 :  0.9122807017543859
Perceptron 의 정답률 :  0.8947368421052632
QuadraticDiscriminantAnalysis 의 정답률 :  0.9385964912280702
RadiusNeighborsClassifier 은 없는 놈!
RandomForestClassifier 의 정답률 :  0.9649122807017544
RidgeClassifier 의 정답률 :  0.956140350877193
RidgeClassifierCV 의 정답률 :  0.9473684210526315
SGDClassifier 의 정답률 :  0.8333333333333334
SVC 의 정답률 :  0.8947368421052632
StackingClassifier 은 없는 놈!
VotingClassifier 은 없는 놈!

======================================================================================2. all_estimators + Kfold
AdaBoostClassifier 의 정답률 :  [0.91208791 0.98901099 0.94505495 0.98901099 0.94505495]
BaggingClassifier 의 정답률 :  [0.96703297 0.93406593 0.97802198 0.9010989  0.93406593]
BernoulliNB 의 정답률 :  [0.51648352 0.65934066 0.64835165 0.6043956  0.69230769]
CalibratedClassifierCV 의 정답률 :  [0.87912088 0.97802198 0.9010989  0.93406593 0.94505495]
CategoricalNB 은 없는 놈!
CheckingClassifier 의 정답률 :  [0. 0. 0. 0. 0.]
ClassifierChain 은 없는 놈!
ComplementNB 의 정답률 :  [0.87912088 0.91208791 0.9010989  0.9010989  0.92307692]
DecisionTreeClassifier 의 정답률 :  [0.89010989 0.94505495 0.93406593 0.87912088 0.96703297]
DummyClassifier 의 정답률 :  [0.62637363 0.6043956  0.47252747 0.6043956  0.54945055]
ExtraTreeClassifier 의 정답률 :  [0.89010989 0.93406593 0.91208791 0.97802198 0.92307692]
ExtraTreesClassifier 의 정답률 :  [0.96703297 0.94505495 0.97802198 0.98901099 0.93406593]
GaussianNB 의 정답률 :  [0.92307692 0.92307692 0.96703297 0.93406593 0.93406593]
GaussianProcessClassifier 의 정답률 :  [0.94505495 0.91208791 0.92307692 0.94505495 0.93406593]
GradientBoostingClassifier 의 정답률 :  [0.95604396 0.95604396 0.92307692 0.95604396 0.94505495]
HistGradientBoostingClassifier 의 정답률 :  [1.         0.91208791 1.         0.93406593 0.96703297]
KNeighborsClassifier 의 정답률 :  [0.95604396 0.93406593 0.94505495 0.91208791 0.91208791]
LabelPropagation 의 정답률 :  [0.47252747 0.41758242 0.34065934 0.49450549 0.28571429]
LabelSpreading 의 정답률 :  [0.41758242 0.43956044 0.3956044  0.41758242 0.34065934]
LinearDiscriminantAnalysis 의 정답률 :  [0.94505495 0.95604396 0.95604396 0.96703297 0.94505495]
LinearSVC 의 정답률 :  [0.93406593 0.74725275 0.83516484 0.95604396 0.85714286]
LogisticRegression 의 정답률 :  [0.91208791 0.95604396 0.91208791 0.93406593 0.96703297]
LogisticRegressionCV 의 정답률 :  [0.93406593 0.94505495 0.93406593 0.97802198 0.96703297]
MLPClassifier 의 정답률 :  [0.93406593 0.92307692 0.89010989 0.92307692 0.98901099]
MultiOutputClassifier 은 없는 놈!
MultinomialNB 의 정답률 :  [0.91208791 0.86813187 0.91208791 0.92307692 0.92307692]
NearestCentroid 의 정답률 :  [0.84615385 0.92307692 0.94505495 0.89010989 0.9010989 ]
NuSVC 의 정답률 :  [0.84615385 0.9010989  0.83516484 0.93406593 0.92307692]
OneVsOneClassifier 은 없는 놈!
OneVsRestClassifier 은 없는 놈!
OutputCodeClassifier 은 없는 놈!
PassiveAggressiveClassifier 의 정답률 :  [0.89010989 0.89010989 0.54945055 0.87912088 0.92307692]
Perceptron 의 정답률 :  [0.87912088 0.9010989  0.73626374 0.97802198 0.86813187]
QuadraticDiscriminantAnalysis 의 정답률 :  [0.95604396 0.96703297 0.94505495 0.93406593 0.98901099]
RadiusNeighborsClassifier 은 없는 놈!
RandomForestClassifier 의 정답률 :  [0.97802198 0.94505495 0.94505495 0.94505495 0.95604396]
RidgeClassifier 의 정답률 :  [0.93406593 0.94505495 0.93406593 0.97802198 0.94505495]
RidgeClassifierCV 의 정답률 :  [0.93406593 0.95604396 0.94505495 0.96703297 0.97802198]
SGDClassifier 의 정답률 :  [0.73626374 0.69230769 0.91208791 0.91208791 0.93406593]
SVC 의 정답률 :  [0.91208791 0.93406593 0.91208791 0.95604396 0.85714286]
StackingClassifier 은 없는 놈!
VotingClassifier 은 없는 놈!
'''