from tensorflow.keras.models import Sequential          #순차모델
from tensorflow.keras.models import Model               #함수모델
from tensorflow.keras.layers import Dense               #레이어   
from tensorflow.keras.layers import Input               #입력레이어
from tensorflow.keras.layers import concatenate         #병합모델 병합레이어
from tensorflow.keras.datasets import boston_hosing     #keras 데이터로드
from tensorflow.keras.callbacks import EarlyStopping    #EarlyStopping

from sklearn.metrics import mean_squared_error          #평가지표
from sklearn.metrics import r2_score                    #평가지표
from sklearn.model_selection import train_test_split    #데이터분리
from sklearn.datasets import load_boston                #sklearn 데이터로드
from sklearn.preprocessing import MinMaxScaler          #데이터 전처리
