from sklearn.datasets import load_diabetes
from sklearn.utils.testing import all_estimators
from sklearn. model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

dataset = load_diabetes()

x = dataset.data
y = dataset.target 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 66)

allAlgorithms = all_estimators(type_filter = 'regressor')

for (name, algorithm) in allAlgorithms:

    try:
        model = algorithm()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률      : ', r2_score(y_test, y_pred))

    except:
        print(name, '은 없는 놈!')


'''
Tensorflow            :
Conv1D모델 r2 :  0.6436679568820876

ARDRegression 의 정답률      :  0.4987483503692143
AdaBoostRegressor 의 정답률      :  0.3511948822938863
BaggingRegressor 의 정답률      :  0.27228873546153953
BayesianRidge 의 정답률      :  0.5008218932350129
CCA 의 정답률      :  0.48696409064967594
DecisionTreeRegressor 의 정답률      :  -0.25602652992426633
DummyRegressor 의 정답률      :  -0.00015425885559339214
ElasticNet 의 정답률      :  0.008101269711286885
ElasticNetCV 의 정답률      :  0.43071557917754755
ExtraTreeRegressor 의 정답률      :  -0.49534915161393434
ExtraTreesRegressor 의 정답률      :  0.3859155333947639
GammaRegressor 의 정답률      :  0.005812599388535289
GaussianProcessRegressor 의 정답률      :  -5.636096407912189
GeneralizedLinearRegressor 의 정답률      :  0.005855247171688949
GradientBoostingRegressor 의 정답률      :  0.3906355971871046
HistGradientBoostingRegressor 의 정답률      :  0.28899497703380905
HuberRegressor 의 정답률      :  0.5033459728718326
IsotonicRegression 은 없는 놈!
KNeighborsRegressor 의 정답률      :  0.3968391279034368
KernelRidge 의 정답률      :  -3.3847644323549924
Lars 의 정답률      :  0.49198665214641635
LarsCV 의 정답률      :  0.5010892359535759
Lasso 의 정답률      :  0.3431557382027084
LassoCV 의 정답률      :  0.49757816595208426
LassoLars 의 정답률      :  0.36543887418957965
LassoLarsCV 의 정답률      :  0.495194279067827
LassoLarsIC 의 정답률      :  0.4994051517531072
LinearRegression 의 정답률      :  0.5063891053505036
LinearSVR 의 정답률      :  -0.33470258280275056
MLPRegressor 의 정답률      :  -2.9329103865387514
MultiOutputRegressor 은 없는 놈!
MultiTaskElasticNet 은 없는 놈!
MultiTaskElasticNetCV 은 없는 놈!
MultiTaskLasso 은 없는 놈!
MultiTaskLassoCV 은 없는 놈!
NuSVR 의 정답률      :  0.14471275169122277
OrthogonalMatchingPursuit 의 정답률      :  0.3293449115305741
OrthogonalMatchingPursuitCV 의 정답률      :  0.44354253337919747
PLSCanonical 의 정답률      :  -0.975079227792292
PLSRegression 의 정답률      :  0.4766139460349792
PassiveAggressiveRegressor 의 정답률      :  0.43858743762665575
PoissonRegressor 의 정답률      :  0.32989738735884344
RANSACRegressor 의 정답률      :  0.3358670401999996
RadiusNeighborsRegressor 의 정답률      :  -0.00015425885559339214
RandomForestRegressor 의 정답률      :  0.37378550312103587
RegressorChain 은 없는 놈!
Ridge 의 정답률      :  0.40936668956159705
RidgeCV 의 정답률      :  0.49525463889305044
SGDRegressor 의 정답률      :  0.39332528117552257
SVR 의 정답률      :  0.14331518075345895
StackingRegressor 은 없는 놈!
TheilSenRegressor 의 정답률      :  0.507289701265205           ******
TransformedTargetRegressor 의 정답률      :  0.5063891053505036
TweedieRegressor 의 정답률      :  0.005855247171688949
VotingRegressor 은 없는 놈!
_SigmoidCalibration 은 없는 놈!
'''