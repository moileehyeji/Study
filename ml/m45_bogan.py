# 결측치 처리방법은 모두 시도해봐야 한다.
# 평균치, 중위값, 보간법, 상단값(frontfill), 하단값(backfill)
# bogan : 결측치처리법
#   이상치라면? 이상치 자리를 nan으로 바꾼후 보간법
#   linearRegressor방식
#   결측치를 제외하고 데이터를 구성
#   model.predict([결측치])   -> 결측치 예측

from pandas import DataFrame, Series
from datetime import datetime
import numpy as np
import pandas as pd

datestrs = ['3/1/2021', '3/2/2021', '3/3/2021', '3/4/2021', '3/5/2021']
dates = pd.to_datetime(datestrs)
print(dates)
print('-----------------------------------------')


#---------------------------------------------결측치 보간 전
ts = Series([1, np.nan, np.nan, 8, 10], index=dates)    #np.nan:결측치
print(ts)
# 2021-03-01     1.0
# 2021-03-02     NaN    *1과 8사이 예상
# 2021-03-03     NaN    *1과 8사이 예상
# 2021-03-04     8.0
# 2021-03-05    10.0
# dtype: float64
#---------------------------------------------결측치 보간 후 interpolate
ts_intp_linear = ts.interpolate()
print(ts_intp_linear)
# 2021-03-01     1.000000
# 2021-03-02     3.333333
# 2021-03-03     5.666667
# 2021-03-04     8.000000
# 2021-03-05    10.000000
# dtype: float64