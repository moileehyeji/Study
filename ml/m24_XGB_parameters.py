'''
xgb.XGBClassifier(
    
    # General Parameter
    booster='gbtree' # 트리,회귀(gblinear) 트리가 항상 
                     # 더 좋은 성능을 내기 때문에 수정할 필요없다고한다.
    
    silent=True  # running message출력안한다.
                 # 모델이 적합되는 과정을 이해하기위해선 False으로한다.
    
    min_child_weight=10   # 값이 높아지면 under-fitting 되는 
                          # 경우가 있다. CV를 통해 튜닝되어야 한다.
    
    max_depth=8     # 트리의 최대 깊이를 정의함. 
                    # 루트에서 가장 긴 노드의 거리.
                    # 8이면 중요변수에서 결론까지 변수가 9개거친다.
                    # Typical Value는 3-10. 
    
    gamma =0    # 노드가 split 되기 위한 loss function의 값이
                # 감소하는 최소값을 정의한다. gamma 값이 높아질 수록 
                # 알고리즘은 보수적으로 변하고, loss function의 정의
                #에 따라 적정값이 달라지기때문에 반드시 튜닝.
    
    nthread =4    # XGBoost를 실행하기 위한 병렬처리(쓰레드)
                  #갯수. 'n_jobs' 를 사용해라.
    
    colsample_bytree=0.8   # 트리를 생성할때 훈련 데이터에서 
                           # 변수를 샘플링해주는 비율. 보통0.6~0.9
    
    colsample_bylevel=0.9  # 트리의 레벨별로 훈련 데이터의 
                           #변수를 샘플링해주는 비율. 보통0.6~0.9
    
    n_estimators =(int)   #부스트트리의 양
                          # 트리의 갯수. 
    
    objective = 'reg:linear','binary:logistic','multi:softmax',
                'multi:softprob'  # 4가지 존재.
            # 회귀 경우 'reg', binary분류의 경우 'binary',
            # 다중분류경우 'multi'- 분류된 class를 return하는 경우 'softmax'
            # 각 class에 속할 확률을 return하는 경우 'softprob'
    
    random_state =  # random number seed.
                    # seed 와 동일.
)




XGBClassifier.fit(
    
    X (array_like)     # Feature matrix ( 독립변수)
                       # X_train
    
    Y (array)          # Labels (종속변수)
                       # Y_train
    
    eval_set           # 빨리 끝나기 위해 검증데이터와 같이써야한다.  
                       # =[(X_train,Y_train),(X_vld, Y_vld)]
 
    eval_metric = 'rmse','error','mae','logloss','merror',
                'mlogloss','auc'  
              # validation set (검증데이터)에 적용되는 모델 선택 기준.
              # 평가측정. 
              # 회귀 경우 rmse ,  분류 -error   이외의 옵션은 함수정의
    
    early_stopping_rounds=100,20
              # 100번,20번 반복동안 최대화 되지 않으면 stop
)
    
​'''
# [출처] 파이썬 Scikit-Learn형식 XGBoost 파라미터|작성자 현무