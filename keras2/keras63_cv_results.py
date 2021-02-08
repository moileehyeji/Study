# keras61_1_hyperParameter 복사
# model.cv_results_

import numpy as np 
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 1. 데이터/ 전처리
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape)
print(y_test.shape)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

# 2. 모델
def build_model(drop=0.5,optimizer='adam'):
    inputs = Input(shape = (28*28),name = 'input')
    x = Dense(512,activation='relu',name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256,activation='relu',name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128,activation='relu',name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10,activation='softmax',name = 'outputs')(x)
    model = Model(inputs=inputs,outputs=outputs)
    model.compile(loss = 'categorical_crossentropy',optimizer = optimizer,metrics = ['acc'])
    return model

def create_hyperparameter() : 
    batchs = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    return {'batch_size' : batchs, 'optimizer' : optimizers, 'drop':dropout}   

hyperparameters = create_hyperparameter()
model = build_model()

#===========================================================================================래핑 :  KerasClassifier
# TypeError: If no scoring is specified, the estimator passed should have a 'score' method. 
# The estimator <tensorflow.python.keras.engine.functional.Functional object at 0x000001AE964FBE80> does not.
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose = 1)

search = RandomizedSearchCV(model2, hyperparameters, cv=3)
# search = GridSearchCV(model2, hyperparameters, cv=3)


search.fit(x_train, y_train, verbose=1)

acc = search.score(x_test, y_test) 
print(search.best_params_)      # 최적의 파라미터 값 출력
print(search.best_estimator_)   
print(search.best_score_)       # 최고의 점수
print('최종스코어 : ', acc)     # 최종스코어 :  0.9638000130653381

print('cv_results_ : ', search.cv_results_)


'''
===========================================================RandomizedSearchCV : 
cv_results_ :  {'mean_fit_time': array([1.76259605, 1.8335743 , 1.86618622, 6.69880589, 3.84874892,
       1.58857187, 2.06404757, 1.59146094, 5.87501097, 3.17180697]), 'std_fit_time': array([0.06106649, 0.03586776, 0.05697003, 0.1067937 , 0.08612877,
       0.06914108, 0.0705529 , 0.06972235, 0.07429396, 0.06987598]), 'mean_score_time': array([0.5345273 , 0.5864923 , 0.46145264, 1.94209679, 0.98626701,
       0.47901082, 0.55632114, 0.47804054, 1.84941419, 0.87034146]), 'std_score_time': array([0.04201615, 0.00444119, 0.00684951, 0.0301694 , 0.01266734,
       0.00503948, 0.00418085, 0.0041714 , 0.01260282, 0.0017721 ]), 'param_optimizer': masked_array(data=['adadelta', 'adam', 'rmsprop', 'rmsprop', 'rmsprop',
                   'adam', 'rmsprop', 'adam', 'adadelta', 'rmsprop'],
             mask=[False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object), 'param_drop': masked_array(data=[0.1, 0.3, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1],
             mask=[False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object), 'param_batch_size': masked_array(data=[50, 40, 50, 10, 20, 50, 40, 50, 10, 30],
             mask=[False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object), 'params': [{'optimizer': 'adadelta', 'drop': 0.1, 'batch_size': 50}, {'optimizer': 'adam', 'drop': 0.3, 'batch_size': 40}, {'optimizer': 'rmsprop', 'drop': 0.3, 'batch_size': 50}, {'optimizer': 'rmsprop', 'drop': 0.3, 'batch_size': 10}, {'optimizer': 'rmsprop', 'drop': 0.2, 'batch_size': 20}, {'optimizer': 'adam', 'drop': 0.2, 'batch_size': 50}, {'optimizer': 'rmsprop', 'drop': 0.1, 'batch_size': 40}, {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 50}, {'optimizer': 'adadelta', 'drop': 0.1, 'batch_size': 10}, {'optimizer': 'rmsprop', 'drop': 0.1, 'batch_size': 30}], 'split0_test_score': array([0.22115   , 0.95920002, 0.95655   , 0.94875002, 0.95324999,
       0.96044999, 0.95875001, 0.96375   , 0.37360001, 0.96095002]), 'split1_test_score': array([0.25725001, 0.9551    , 0.949     , 0.94679999, 0.95450002,
       0.95560002, 0.95015001, 0.95880002, 0.40454999, 0.95275003]), 'split2_test_score': array([0.32624999, 0.95520002, 0.95679998, 0.9483    , 0.95424998,
       0.95649999, 0.95955002, 0.95335001, 0.50019997, 0.95765001]), 'mean_test_score': array([0.26821666, 0.95650001, 0.95411666, 0.94795001, 0.954     ,
       0.95751667, 0.95615002, 0.95863334, 0.42611666, 0.95711668]), 'std_test_score': array([0.04360201, 0.00190963, 0.00361947, 0.00083368, 0.00054007,
       0.00210646, 0.00425519, 0.00424742, 0.0538871 , 0.00336881]), 'rank_test_score': array([10,  4,  6,  8,  7,  2,  5,  1,  9,  3])}

===========================================================GridSearchCV : 
cv_results_ :  
    {'mean_fit_time': array([6.80414693, 5.81048656, 5.90585971, 6.53819331, 5.55487879,
       5.7687703 , 6.55284802, 5.56006519, 5.69299046, 3.85948292,
       3.18871315, 3.37832816, 3.86999011, 3.16373007, 3.43038789,
       3.86875478, 3.15995916, 3.40550733, 3.1287237 , 2.70519622,
       2.79443574, 3.07318234, 2.65406958, 2.69485879, 3.02275109,
       2.74671022, 2.68792733, 2.08373968, 1.80393887, 1.86547367,
       2.02852146, 1.78023982, 1.87245893, 2.01511335, 1.70148849,
       1.80423705, 1.86640414, 1.60605454, 1.61563579, 1.78510348,
       1.52582423, 1.6805586 , 1.86555179, 1.50572451, 1.67665108]), 
       'std_fit_time': array([0.08327126, 0.13667632, 0.02689215, 0.02078803, 0.09681952,
       0.03983677, 0.05894172, 0.07446542, 0.06055186, 0.07425895,
       0.09253856, 0.1003973 , 0.11802843, 0.07786677, 0.12163976,
       0.0911733 , 0.0989658 , 0.05322751, 0.07121041, 0.07746687,
       0.08387436, 0.07944792, 0.0384155 , 0.00480862, 0.0132542 ,
       0.13188547, 0.03353849, 0.12022493, 0.15200789, 0.09956523,
       0.01979272, 0.09033077, 0.10002442, 0.00834225, 0.02753384,
       0.0176199 , 0.1418881 , 0.14404441, 0.01875343, 0.00902081,
       0.0327974 , 0.15908907, 0.12417649, 0.0425664 , 0.16284347]), 'mean_score_time': array([1.96444829, 1.87947679, 1.90613127, 1.86228553, 1.83255823,
       1.82522051, 1.82713374, 1.94219073, 1.93227132, 0.97792141,
       0.98271497, 0.9776725 , 0.96690162, 0.97820067, 0.97283189,
       0.97681181, 0.96694938, 0.97784742, 0.86449941, 0.85712949,
       0.89541101, 0.91095161, 0.90864833, 0.84436758, 0.90923691,
       0.8409218 , 0.84544889, 0.54368464, 0.55757268, 0.55362042,
       0.54686689, 0.54668419, 0.58362571, 0.54597696, 0.63496987,
       0.55726258, 0.4622492 , 0.48896313, 0.487837  , 0.54805366,
       0.46860878, 0.46818693, 0.46286186, 0.47351074, 0.47209112]), 'std_score_time': array([0.02154222, 0.02522158, 0.03946837, 0.05277764, 0.019986  ,
       0.00725307, 0.00399427, 0.14415447, 0.1282579 , 0.00860757,
       0.01581107, 0.00579826, 0.00823669, 0.00296715, 0.01395514,
       0.00615512, 0.0032918 , 0.00752391, 0.02799836, 0.00455575,
       0.07945113, 0.07720664, 0.107304  , 0.00437371, 0.12558839,
       0.00458125, 0.0264514 , 0.00280097, 0.00652287, 0.00809917,
       0.01017427, 0.00132118, 0.02159832, 0.00347032, 0.10868907,
       0.00239867, 0.00320689, 0.01328039, 0.02185019, 0.12101401,
       0.00856856, 0.00254327, 0.00415068, 0.0020545 , 0.00305991]), 'param_batch_size': masked_array(data=[10, 10, 10, 10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20,
                   20, 20, 20, 20, 30, 30, 30, 30, 30, 30, 30, 30, 30, 40,
                   40, 40, 40, 40, 40, 40, 40, 40, 50, 50, 50, 50, 50, 50,
                   50, 50, 50],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False],
       fill_value='?',
            dtype=object), 'param_drop': masked_array(data=[0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.1, 0.1,
                   0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.2,
                   0.2, 0.2, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2,
                   0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3,
                   0.3],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False],
       fill_value='?',
            dtype=object), 'param_optimizer': masked_array(data=['rmsprop', 'adam', 'adadelta', 'rmsprop', 'adam',
                   'adadelta', 'rmsprop', 'adam', 'adadelta', 'rmsprop',
                   'adam', 'adadelta', 'rmsprop', 'adam', 'adadelta',
                   'rmsprop', 'adam', 'adadelta', 'rmsprop', 'adam',
                   'adadelta', 'rmsprop', 'adam', 'adadelta', 'rmsprop',
                   'adam', 'adadelta', 'rmsprop', 'adam', 'adadelta',
                   'rmsprop', 'adam', 'adadelta', 'rmsprop', 'adam',
                   'adadelta', 'rmsprop', 'adam', 'adadelta', 'rmsprop',
                   'adam', 'adadelta', 'rmsprop', 'adam', 'adadelta'],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False],
       fill_value='?',
            dtype=object), 'params': [{'batch_size': 10, 'drop': 0.1, 'optimizer': 'rmsprop'}, {'batch_size': 10, 'drop': 0.1, 'optimizer': 'adam'}, {'batch_size': 10, 'drop': 0.1, 'optimizer': 'adadelta'}, {'batch_size': 10, 'drop': 0.2, 'optimizer': 'rmsprop'}, {'batch_size': 10, 'drop': 0.2, 'optimizer': 'adam'}, {'batch_size': 10, 'drop': 0.2, 'optimizer': 'adadelta'}, {'batch_size': 10, 'drop': 0.3, 'optimizer': 'rmsprop'}, {'batch_size': 10, 'drop': 0.3, 'optimizer': 'adam'}, {'batch_size': 10, 'drop': 0.3, 'optimizer': 'adadelta'}, {'batch_size': 20, 'drop': 0.1, 'optimizer': 'rmsprop'}, {'batch_size': 20, 'drop': 0.1, 'optimizer': 'adam'}, {'batch_size': 20, 'drop': 0.1, 'optimizer': 'adadelta'}, {'batch_size': 20, 'drop': 0.2, 'optimizer': 'rmsprop'}, {'batch_size': 20, 'drop': 0.2, 'optimizer': 'adam'}, {'batch_size': 20, 'drop': 0.2, 'optimizer': 'adadelta'}, {'batch_size': 20, 'drop': 0.3, 'optimizer': 'rmsprop'}, {'batch_size': 20, 'drop': 0.3, 'optimizer': 'adam'}, {'batch_size': 20, 'drop': 0.3, 'optimizer': 'adadelta'}, {'batch_size': 30, 'drop': 0.1, 'optimizer': 'rmsprop'}, {'batch_size': 30, 'drop': 0.1, 'optimizer': 'adam'}, {'batch_size': 30, 'drop': 0.1, 'optimizer': 'adadelta'}, {'batch_size': 30, 'drop': 0.2, 'optimizer': 'rmsprop'}, {'batch_size': 30, 'drop': 0.2, 'optimizer': 'adam'}, {'batch_size': 30, 'drop': 0.2, 'optimizer': 'adadelta'}, {'batch_size': 30, 'drop': 0.3, 'optimizer': 'rmsprop'}, {'batch_size': 30, 'drop': 0.3, 'optimizer': 'adam'}, {'batch_size': 30, 'drop': 0.3, 
'optimizer': 'adadelta'}, {'batch_size': 40, 'drop': 0.1, 'optimizer': 'rmsprop'}, {'batch_size': 40, 'drop': 0.1, 'optimizer': 'adam'}, {'batch_size': 40, 'drop': 0.1, 'optimizer': 'adadelta'}, {'batch_size': 40, 'drop': 0.2, 'optimizer': 'rmsprop'}, {'batch_size': 40, 'drop': 0.2, 'optimizer': 'adam'}, {'batch_size': 40, 'drop': 0.2, 'optimizer': 'adadelta'}, {'batch_size': 40, 'drop': 0.3, 'optimizer': 'rmsprop'}, {'batch_size': 40, 'drop': 0.3, 'optimizer': 'adam'}, {'batch_size': 40, 'drop': 0.3, 'optimizer': 'adadelta'}, {'batch_size': 50, 'drop': 0.1, 'optimizer': 'rmsprop'}, {'batch_size': 50, 'drop': 0.1, 'optimizer': 'adam'}, {'batch_size': 50, 'drop': 0.1, 'optimizer': 'adadelta'}, {'batch_size': 50, 'drop': 0.2, 'optimizer': 'rmsprop'}, {'batch_size': 50, 'drop': 0.2, 'optimizer': 'adam'}, {'batch_size': 50, 'drop': 0.2, 'optimizer': 'adadelta'}, {'batch_size': 50, 'drop': 0.3, 'optimizer': 'rmsprop'}, {'batch_size': 50, 'drop': 0.3, 'optimizer': 'adam'}, {'batch_size': 50, 'drop': 0.3, 'optimizer': 'adadelta'}], 'split0_test_score': array([0.92879999, 0.95959997, 0.38870001, 0.95684999, 0.95719999,
       0.36695001, 0.94835001, 0.94765002, 0.36129999, 0.9598    ,
       0.95604998, 0.33860001, 0.95034999, 0.96100003, 0.2406    ,
       0.95574999, 0.94645   , 0.29155001, 0.96149999, 0.94730002,
       0.3141    , 0.96004999, 0.95805001, 0.25444999, 0.96069998,
       0.95644999, 0.18664999, 0.95945001, 0.96415001, 0.23459999,
       0.96039999, 0.958     , 0.2798    , 0.95740002, 0.95475   ,
       0.1455    , 0.95230001, 0.96289998, 0.25725001, 0.95695001,
       0.958     , 0.21035001, 0.95209998, 0.9569    , 0.1015    ]), 'split1_test_score': array([0.94545001, 0.94340003, 0.34615001, 0.94884998, 0.95679998,
       0.34305   , 0.94959998, 0.95214999, 0.25330001, 0.95244998,
       0.94115001, 0.43605   , 0.94889998, 0.95359999, 0.31900001,
       0.94929999, 0.95305002, 0.28255001, 0.95789999, 0.9569    ,
       0.22135   , 0.95165002, 0.95560002, 0.30135   , 0.94185001,
       0.94499999, 0.1715    , 0.95744997, 0.95749998, 0.23285   ,
       0.95545   , 0.95569998, 0.21795   , 0.94935   , 0.95225   ,
       0.234     , 0.95894998, 0.95340002, 0.2344    , 0.95055002,
       0.95649999, 0.27630001, 0.94440001, 0.95335001, 0.12715   ]), 'split2_test_score': array([0.95279998, 0.95095003, 0.4179    , 0.94924998, 0.9526    ,
       0.32325   , 0.94924998, 0.9454    , 0.30915001, 0.96284997,
       0.95310003, 0.37445   , 0.95959997, 0.96104997, 0.3423    ,
       0.9551    , 0.95254999, 0.2017    , 0.958     , 0.96079999,
       0.34175   , 0.95740002, 0.95894998, 0.2052    , 0.9551    ,
       0.95104998, 0.18125001, 0.95885003, 0.95810002, 0.19915   ,
       0.95994997, 0.95670003, 0.18265   , 0.95240003, 0.95354998,
       0.1996    , 0.96130002, 0.96034998, 0.28395   , 0.94875002,
       0.95560002, 0.22220001, 0.95359999, 0.95550001, 0.14605001]), 'mean_test_score': array([0.94234999, 0.95131667, 0.38425001, 0.95164998, 0.95553333,
       0.34441667, 0.94906666, 0.9484    , 0.30791667, 0.95836665,
       0.9501    , 0.38303334, 0.95294998, 0.95855   , 0.30063334,
       0.95338333, 0.95068334, 0.25860001, 0.95913333, 0.955     ,
       0.2924    , 0.95636668, 0.95753334, 0.25366666, 0.95254999,
       0.95083332, 0.1798    , 0.95858334, 0.95991667, 0.2222    ,
       0.95859998, 0.9568    , 0.2268    , 0.95305002, 0.95351666,
       0.19303333, 0.95751667, 0.95888333, 0.25853334, 0.95208335,
       0.95670001, 0.23628334, 0.95003333, 0.95525   , 0.1249    ]), 'std_test_score': array([0.01004017, 0.00661868, 0.02946033, 0.00368059, 0.00208059,
       0.01786661, 0.00052651, 0.00280624, 0.04409943, 0.00436508,
       0.00644217, 0.04024409, 0.00473937, 0.00350024, 0.04350267,
       0.00289952, 0.00300038, 0.0404018 , 0.00167398, 0.00567273,
       0.05149244, 0.00350625, 0.00141557, 0.03925698, 0.00790389,
       0.00467695, 0.00626937, 0.000838  , 0.00300343, 0.01631446,
       0.00223494, 0.00094164, 0.04015198, 0.00331839, 0.00102089,
       0.03642712, 0.00381146, 0.00401461, 0.02024905, 0.00351883,
       0.00098994, 0.02870663, 0.00403015, 0.00146002, 0.01825692]), 'rank_test_score': array([30, 23, 31, 22, 13, 33, 28, 29, 34,  7, 26, 32, 19,  6, 35, 17, 25,
       37,  2, 15, 36, 12,  8, 39, 20, 24, 44,  5,  1, 42,  4, 10, 41, 18,
       16, 43,  9,  3, 38, 21, 11, 40, 27, 14, 45])}
'''



