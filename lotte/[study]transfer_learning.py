[전이황님 전이학습 모델 판단기준]

https://hoya012.github.io/blog/EfficientNet-review/

각 모델의 [ImageNet에 대한 실험 결과] 참고
1. Params
2. Accuracy 

* 파라미터가 크고 acc가 상대적으로 크지 않으면 비효율적이다.
* 데이터가 적으면 파라미터가 많은 모델, 많으면 적은 모델을 선택해서 테스트 해보자.
* 각 모델에 최적화된 activation, optimizer확인하여 테스트하자.
* 사전훈련데이터의 크기를 맞춰서 테스트 해보자.
* 데이터가 적으면 통상적으로 efficientnets 효율적?
* 데이터가 많으면 통상적으로 mobilenets 효율적?