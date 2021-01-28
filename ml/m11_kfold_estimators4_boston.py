# m11_kfold_estimators1 복사
# all_estimators + Kfold, cross_val_score

# 모델 비교

# all_estimators 

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_boston
import warnings

warnings.filterwarnings('ignore')

dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 66)

kfold = KFold(n_splits=5, shuffle=True)


# ====================================================all_estimators : sklearn에서 제공하는 회귀형모델의 전체를 훈련
allAlgorithms = all_estimators(type_filter = 'regressor')

for (name, algorithm) in allAlgorithms:

    try:
        model = algorithm()

        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        print(name, '의 정답률 : ', scores)

        # model.fit(x_train, y_train)
        # y_pred = model.predict(x_test)

    except:
        # continue
        print(name, '은 없는 놈!')

import sklearn
print(sklearn.__version__)      # 0.23.2


'''

Tensorflow            :
CNN모델 r2 :  0.9462232137123261

======================================================================================1. all_estimators

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

======================================================================================2. all_estimators + Kfold
ARDRegression 의 정답률 :  [0.70563478 0.67718385 0.67270921 0.73447026 0.64046251]
AdaBoostRegressor 의 정답률 :  [0.78118054 0.85252223 0.85650254 0.79604259 0.70611058]
BaggingRegressor 의 정답률 :  [0.90199228 0.85434195 0.88877068 0.50354513 0.84846137]
BayesianRidge 의 정답률 :  [0.67244956 0.74320714 0.55371091 0.65191788 0.69071234]
CCA 의 정답률 :  [0.76625086 0.61229115 0.66915916 0.64949014 0.59881218]
DecisionTreeRegressor 의 정답률 :  [0.70090478 0.81067982 0.19620762 0.85564247 0.79032404]
DummyRegressor 의 정답률 :  [-0.00236258 -0.01884765 -0.07308115 -0.03400897 -0.08546896]
ElasticNet 의 정답률 :  [0.64804554 0.68762364 0.59682949 0.66310549 0.64129686]
ElasticNetCV 의 정답률 :  [0.68882456 0.61849209 0.51402174 0.68793809 0.62933079]
ExtraTreeRegressor 의 정답률 :  [0.8184351  0.64259256 0.75872688 0.28785227 0.67789618]
ExtraTreesRegressor 의 정답률 :  [0.86362417 0.88818893 0.91293169 0.84454386 0.85686349]
GammaRegressor 의 정답률 :  [-1.64093749e-03 -3.10903842e-03 -1.08991942e-05 -1.36608637e-02
 -7.81217606e-04]
GaussianProcessRegressor 의 정답률 :  [-5.96736353 -5.82944374 -6.67413655 -5.70905806 -5.4360859 ]
GeneralizedLinearRegressor 의 정답률 :  [0.63004067 0.70724497 0.63314667 0.65287058 0.54419858]
GradientBoostingRegressor 의 정답률 :  [0.91599853 0.86372954 0.87789132 0.84594204 0.73152122]
HistGradientBoostingRegressor 의 정답률 :  [0.92235866 0.9128818  0.69689037 0.7642945  0.86405863]
HuberRegressor 의 정답률 :  [0.55797092 0.70560869 0.346046   0.61723955 0.70107732]
IsotonicRegression 의 정답률 :  [nan nan nan nan nan]
KNeighborsRegressor 의 정답률 :  [0.51922275 0.54170558 0.5451097  0.27769513 0.45227113]
KernelRidge 의 정답률 :  [0.7869654  0.27510954 0.69579518 0.72685503 0.68397728]
Lars 의 정답률 :  [0.68357653 0.72808258 0.65075286 0.6808563  0.67897225]
LarsCV 의 정답률 :  [0.75266181 0.68667111 0.60403874 0.77890285 0.57113706]
Lasso 의 정답률 :  [0.60998549 0.67542058 0.6085871  0.69310108 0.55189231]
LassoCV 의 정답률 :  [0.67639449 0.66633133 0.61501165 0.68161323 0.69495553]
LassoLars 의 정답률 :  [-0.00556783 -0.00497878 -0.04155039 -0.15781311 -0.0060277 ]
LassoLarsCV 의 정답률 :  [0.68147723 0.65849921 0.79703991 0.66893791 0.63725999]
LassoLarsIC 의 정답률 :  [0.66172616 0.75919907 0.76587939 0.58734669 0.660492  ]
LinearRegression 의 정답률 :  [0.69659439 0.78610113 0.60137075 0.71865027 0.55946186]
LinearSVR 의 정답률 :  [0.36672891 0.41304319 0.56316924 0.6468179  0.32110237]
MLPRegressor 의 정답률 :  [0.51203917 0.50553227 0.57703923 0.49411964 0.54074315]
MultiOutputRegressor 은 없는 놈!
MultiTaskElasticNet 의 정답률 :  [nan nan nan nan nan]
MultiTaskElasticNetCV 의 정답률 :  [nan nan nan nan nan]
MultiTaskLasso 의 정답률 :  [nan nan nan nan nan]
MultiTaskLassoCV 의 정답률 :  [nan nan nan nan nan]
NuSVR 의 정답률 :  [0.19942296 0.18820009 0.30032178 0.15238728 0.22783636]
OrthogonalMatchingPursuit 의 정답률 :  [0.59716741 0.54546966 0.44213356 0.54073044 0.51501186]
OrthogonalMatchingPursuitCV 의 정답률 :  [0.58767704 0.40123382 0.70424952 0.70607314 0.72750705]
PLSCanonical 의 정답률 :  [-1.81120591 -2.25840997 -2.1139571  -1.98522098 -2.75874529]
PLSRegression 의 정답률 :  [0.73961465 0.56385342 0.72595506 0.70540112 0.5672698 ]
PassiveAggressiveRegressor 의 정답률 :  [ 0.1297087   0.24828615  0.10396916 -0.1010457  -2.84276483]
PoissonRegressor 의 정답률 :  [0.75921541 0.69434352 0.70452911 0.70177804 0.79639628]
RANSACRegressor 의 정답률 :  [0.54956719 0.41204909 0.29425312 0.50038612 0.48105175]
RadiusNeighborsRegressor 은 없는 놈!
RandomForestRegressor 의 정답률 :  [0.79536108 0.8599195  0.8253182  0.9201452  0.81166685]
RegressorChain 은 없는 놈!
Ridge 의 정답률 :  [0.67143322 0.74218588 0.70932423 0.60333899 0.7544004 ]
RidgeCV 의 정답률 :  [0.69462623 0.70446549 0.75728687 0.59588664 0.65372073]
SGDRegressor 의 정답률 :  [-5.61569676e+25 -7.17459378e+25 -1.91169080e+27 -2.11810395e+26
 -6.10186944e+26]
SVR 의 정답률 :  [ 0.25487852  0.14430334  0.23535768  0.22747841 -0.03375848]
StackingRegressor 은 없는 놈!
TheilSenRegressor 의 정답률 :  [0.56518947 0.4922234  0.81346107 0.71712837 0.5873599 ]
TransformedTargetRegressor 의 정답률 :  [0.7215823  0.62073341 0.77869747 0.63221506 0.7056588 ]
TweedieRegressor 의 정답률 :  [0.53196034 0.69235841 0.60755037 0.70798386 0.58612619]
VotingRegressor 은 없는 놈!
_SigmoidCalibration 의 정답률 :  [nan nan nan nan nan]

'''