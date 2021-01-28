# all_estimators 

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_boston
import warnings

warnings.filterwarnings('ignore')

dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 66)


# ====================================================all_estimators : sklearn에서 제공하는 회귀형모델의 전체를 훈련
allAlgorithms = all_estimators(type_filter = 'regressor')

for (name, algorithm) in allAlgorithms:

    try:
        model = algorithm()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률 : ', r2_score(y_test, y_pred))

    except:
        # continue
        print(name, '은 없는 놈!')

import sklearn
print(sklearn.__version__)      # 0.23.2


'''

Tensorflow            :
CNN모델 r2 :  0.9462232137123261


ARDRegression 의 정답률 :  0.8012569266997974
AdaBoostRegressor 의 정답률 :  0.8978370620789664
BaggingRegressor 의 정답률 :  0.9201200133100753
BayesianRidge 의 정답률 :  0.7937918622384766
CCA 의 정답률 :  0.7913477184424629
DecisionTreeRegressor 의 정답률 :  0.799559193450648
DummyRegressor 의 정답률 :  -0.0005370164400797517
ElasticNet 의 정답률 :  0.7338335519267194
ElasticNetCV 의 정답률 :  0.7167760356856181
ExtraTreeRegressor 의 정답률 :  0.8659954943720589
ExtraTreesRegressor 의 정답률 :  0.9354181636737243
GammaRegressor 의 정답률 :  -0.0005370164400797517
GaussianProcessRegressor 의 정답률 :  -6.073105259620457
GeneralizedLinearRegressor 의 정답률 :  0.7461438417572277
GradientBoostingRegressor 의 정답률 :  0.9463514194242866   ******
HistGradientBoostingRegressor 의 정답률 :  0.9323597806119726
HuberRegressor 의 정답률 :  0.7472151075110175
IsotonicRegression 은 없는 놈!
KNeighborsRegressor 의 정답률 :  0.5900872726222293
KernelRidge 의 정답률 :  0.8333325493719488
Lars 의 정답률 :  0.7746736096721595
LarsCV 의 정답률 :  0.7981576314184016
Lasso 의 정답률 :  0.7240751024070102
LassoCV 의 정답률 :  0.7517507753137198
LassoLars 의 정답률 :  -0.0005370164400797517
LassoLarsCV 의 정답률 :  0.8127604328474287
LassoLarsIC 의 정답률 :  0.8131423868817642
LinearRegression 의 정답률 :  0.8111288663608656
LinearSVR 의 정답률 :  0.7668675128021726
MLPRegressor 의 정답률 :  0.5188880973694936
MultiOutputRegressor 은 없는 놈!
MultiTaskElasticNet 은 없는 놈!
MultiTaskElasticNetCV 은 없는 놈!
MultiTaskLasso 은 없는 놈!
MultiTaskLassoCV 은 없는 놈!
NuSVR 의 정답률 :  0.2594558622083819
OrthogonalMatchingPursuit 의 정답률 :  0.5827617571381449
OrthogonalMatchingPursuitCV 의 정답률 :  0.78617447738729
PLSCanonical 의 정답률 :  -2.2317079741425756
PLSRegression 의 정답률 :  0.8027313142007887
PassiveAggressiveRegressor 의 정답률 :  0.11098736673433474
PoissonRegressor 의 정답률 :  0.8575650836250985
RANSACRegressor 의 정답률 :  0.2195356047173319
RadiusNeighborsRegressor 은 없는 놈!
RandomForestRegressor 의 정답률 :  0.9198873226293084
RegressorChain 은 없는 놈!
Ridge 의 정답률 :  0.8098487632912241
RidgeCV 의 정답률 :  0.8112529186351158
SGDRegressor 의 정답률 :  -7.510298859240489e+25
SVR 의 정답률 :  0.2347467755572229
StackingRegressor 은 없는 놈!
TheilSenRegressor 의 정답률 :  0.7929548376376726
TransformedTargetRegressor 의 정답률 :  0.8111288663608656
TweedieRegressor 의 정답률 :  0.7461438417572277
VotingRegressor 은 없는 놈!
_SigmoidCalibration 은 없는 놈!
'''