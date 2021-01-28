# m11_kfold_estimators1 복사
# all_estimators + Kfold, cross_val_score

# 모델 비교

from sklearn.datasets import load_diabetes
from sklearn.utils.testing import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

dataset = load_diabetes()

x = dataset.data
y = dataset.target 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 66)

kfold = KFold(n_splits=5, shuffle=True)

allAlgorithms = all_estimators(type_filter = 'regressor')

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
Tensorflow            :
Conv1D모델 r2 :  0.6436679568820876

======================================================================================1. all_estimators

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

======================================================================================2. all_estimators + Kfold

ARDRegression 의 정답률 :  [0.49330725 0.57005188 0.48363856 0.40449003 0.44817969]
AdaBoostRegressor 의 정답률 :  [0.56559918 0.5324792  0.4121737  0.2926377  0.32169952]
BaggingRegressor 의 정답률 :  [0.27911406 0.40939064 0.52960375 0.44755865 0.44132138]
BayesianRidge 의 정답률 :  [0.41629245 0.45434838 0.50691127 0.40828151 0.5829841 ]
CCA 의 정답률 :  [0.51084392 0.2613527  0.58734216 0.27103013 0.21656699]
DecisionTreeRegressor 의 정답률 :  [-0.40414744  0.21141191 -0.19381043 -0.09125225  0.11664893]
DummyRegressor 의 정답률 :  [-0.00703593 -0.00880355 -0.00652985 -0.01750625 -0.00382924]
ElasticNet 의 정답률 :  [-0.00714906  0.00843414  0.00414175  0.00833052 -0.02468139]
ElasticNetCV 의 정답률 :  [0.42039992 0.50501948 0.38986244 0.4391717  0.40732253]
ExtraTreeRegressor 의 정답률 :  [-0.18630566  0.0270501  -0.3223861   0.20919725 -0.17985194]
ExtraTreesRegressor 의 정답률 :  [0.45313515 0.46704405 0.58519742 0.55312577 0.30115532]
GammaRegressor 의 정답률 :  [ 0.00074414 -0.00925261 -0.00549923 -0.00062824 -0.00364811]
GaussianProcessRegressor 의 정답률 :  [-10.00117826 -12.73665191 -16.98821611 -20.71287813 -12.43919673]
GeneralizedLinearRegressor 의 정답률 :  [ 0.0026373  -0.06764407 -0.02606812 -0.017757   -0.03775276]
GradientBoostingRegressor 의 정답률 :  [0.46617083 0.39482171 0.4363786  0.54994702 0.35608851]
HistGradientBoostingRegressor 의 정답률 :  [0.32429992 0.50994369 0.37212863 0.42724845 0.4798111 ]
HuberRegressor 의 정답률 :  [0.30300017 0.54992035 0.47103344 0.55688442 0.42422827]
IsotonicRegression 의 정답률 :  [nan nan nan nan nan]
KNeighborsRegressor 의 정답률 :  [0.4513416  0.38360311 0.32813404 0.35355521 0.45557765]
KernelRidge 의 정답률 :  [-4.57514692 -3.59417081 -3.44723924 -2.94001412 -4.11846484]
Lars 의 정답률 :  [0.37242446 0.41696572 0.48816556 0.48008924 0.53164825]
LarsCV 의 정답률 :  [0.58080228 0.39908034 0.4205933  0.50085282 0.49782658]
Lasso 의 정답률 :  [0.30476068 0.3656251  0.40963374 0.30128873 0.32759468]
LassoCV 의 정답률 :  [0.59078915 0.43019894 0.51748858 0.35985608 0.46496033]
LassoLars 의 정답률 :  [0.37321744 0.40226289 0.3760569  0.39676727 0.39136852]
LassoLarsCV 의 정답률 :  [0.53765802 0.53260478 0.40593919 0.46383765 0.5016662 ]
LassoLarsIC 의 정답률 :  [0.4852888  0.40398856 0.50807266 0.36166405 0.57132084]
LinearRegression 의 정답률 :  [0.36144832 0.604527   0.43417696 0.46257968 0.47924384]
LinearSVR 의 정답률 :  [-0.49800891 -0.23767038 -0.70883304 -0.25064855 -0.8057732 ]
MLPRegressor 의 정답률 :  [-3.25119852 -2.4239903  -3.21035428 -4.05687599 -3.2871572 ]
MultiOutputRegressor 은 없는 놈!
MultiTaskElasticNet 의 정답률 :  [nan nan nan nan nan]
MultiTaskElasticNetCV 의 정답률 :  [nan nan nan nan nan]
MultiTaskLasso 의 정답률 :  [nan nan nan nan nan]
MultiTaskLassoCV 의 정답률 :  [nan nan nan nan nan]
NuSVR 의 정답률 :  [0.11415953 0.1209538  0.12212657 0.12951744 0.13769197]
OrthogonalMatchingPursuit 의 정답률 :  [0.17539012 0.25101985 0.21424025 0.35638402 0.39742196]
OrthogonalMatchingPursuitCV 의 정답률 :  [0.42962945 0.28668324 0.53053142 0.47919597 0.58234021]
PLSCanonical 의 정답률 :  [-0.45440716 -0.94937839 -1.52168341 -2.1434491  -1.68178993]
PLSRegression 의 정답률 :  [0.41991613 0.46586271 0.5516907  0.58127392 0.40882733]
PassiveAggressiveRegressor 의 정답률 :  [0.51050851 0.48632037 0.53673967 0.35825434 0.36046801]
PoissonRegressor 의 정답률 :  [0.39132176 0.38082664 0.33790669 0.31228413 0.20982825]
RANSACRegressor 의 정답률 :  [0.37566498 0.42796815 0.1644065  0.17781748 0.18237459]
RadiusNeighborsRegressor 의 정답률 :  [-7.84731066e-03 -1.72585615e-02 -7.45155675e-02 -9.58044709e-05
 -3.05319913e-03]
RandomForestRegressor 의 정답률 :  [0.52614333 0.49388874 0.3016452  0.48657646 0.44308015]
RegressorChain 은 없는 놈!
Ridge 의 정답률 :  [0.42482005 0.33345473 0.42559031 0.39568257 0.38516561]
RidgeCV 의 정답률 :  [0.4482795  0.39842343 0.46589646 0.47422625 0.56191126]
SGDRegressor 의 정답률 :  [0.32956723 0.38360617 0.42599335 0.38019207 0.42164641]
SVR 의 정답률 :  [0.12812875 0.12628901 0.14294549 0.11563903 0.16747455]
StackingRegressor 은 없는 놈!
TheilSenRegressor 의 정답률 :  [0.47762926 0.51024274 0.4809938  0.27086226 0.52741842]
TransformedTargetRegressor 의 정답률 :  [0.46727604 0.56568029 0.48649422 0.52435621 0.30395844]
TweedieRegressor 의 정답률 :  [ 0.00376024  0.00483073  0.0050399   0.00275044 -0.00077376]
VotingRegressor 은 없는 놈!
_SigmoidCalibration 의 정답률 :  [nan nan nan nan nan]
'''