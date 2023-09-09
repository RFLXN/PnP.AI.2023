# 1-1. 훈련 세트와 테스트 세트

## 지도 학습과 비지도 학습

* 지도 학습: 훈련(학습)하기 위해서 데이터와 **정답**이 필요함.    
    이전의 [0-3. 마켓과 머신러닝](https://github.com/RFLXN/PnP.AI.2023/tree/main/doc/0.%20%EB%82%98%EC%9D%98%20%EC%B2%AB%20%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/3.%20%EB%A7%88%EC%BC%93%EA%B3%BC%20%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D)에서 사용했던 k-최근접 이웃은 지도학습의 하나임.    
    지도 학습에서는 데이터를 input, 정답을 target이라는 용어를 이용해서 말하고, 이 두 개를 합쳐서 traning data라고 부른다.
* 비지도 학습: 지도 학습과 다르게 **정답 (target)**이 없이 데이터를 학습함.
* 강화 학습: AI가 만들어 낸 결과에 대한 피드백을 통해서, 다시 AI 자기 자신을 학습시키는 학습 방법. (알파고가 강화학습을 이용했음)

## 훈련 세트와 테스트 세트

* 훈련 세트: 머신러닝에 이용되는 데이터
* 테스트 세트: 학습한 모델의 평가에 사용되는 데이터

머신러닝의 정확한 평가를 위해서는 훈련 세트와 테스트 세트의 데이터가 달라야 한다.    
이를 가장 쉽게 준비하는 방법은 이미 준비된 다수의 훈련 세트 중 일부만을 훈련 세트로 이용하고,    
나머지를 테스트 세트로 이용하는 방법을 이용 할 수 있다.

## 샘플링 편향

훈련 세트, 테스트 세트에서 각각의 데이터(개체)를 샘플이라고 부른다.   
훈련 세트와 테스트 세트에는 서로 다른 target을 의미하는 샘플들이 골고루 섞여있어야 제대로 된 학습/테스트가 가능한데, 이 데이터셋 들이 제대로 섞여있지 않는 것을 sampling bias(샘플링 편향) 이라고 한다.

## 머신러닝 실행해보기

훈련 세트와 테스트 세트를 별개의 데이터로 구분하고, 샘플링 편향을 해결한 채로 머신러닝을 실행해 보자.

우선 데이터 세트를 준비한다.
```python
import numpy

# input을 위한 생선의 길이, 무게의 2차원 배열
fish_data = [[ln, wt] for ln, wt in zip(fish_length, fish_weight)]

# 해답 배열 (도미인지 빙어인지)
fish_target = ([1] * 35) + ([0] * 14)

# python list를 numpy array로 변환
input_data = numpy.array(fish_data)
target_data = numpy.array(fish_target)
```

샘플링 편향을 방지하기 위해, 랜덤한 index를 생성한다.
```python
def shuffle(seed: int, length: int) -> numpy.ndarray:
    numpy.random.seed(seed)
    arr = numpy.arange(length)
    numpy.random.shuffle(arr)

    return arr


index = shuffle(42, len(fish_data))
```

랜덤한 index를 기반으로 훈련 세트와 테스트 세트를 분리한다.

```python
index = shuffle(42, len(fish_data))

train_idx = index[:35]
input_train = input_data[train_idx]
target_train = target_data[train_idx]

test_idx = index[35:]
input_test = input_data[test_idx]
target_test = target_data[test_idx]
```

이제 준비된 훈련 세트와 테스트 세트를 이용해서 이전과 같이 머신러닝을 실행한다.

```python
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()

kn.fit(input_train, target_train)

kn.score(input_test, target_test)
# output: 1.0

kn.predict(input_test)
# output: [0 0 1 0 1 1 1 0 1 1 0 1 1 0]

target_test
# output: [0 0 1 0 1 1 1 0 1 1 0 1 1 0] -> same to predicted result
```

</br></br></br></br>

[다음 (1-2. 데이터 전처리) ->](https://github.com/RFLXN/PnP.AI.2023/tree/main/doc/1.%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EB%8B%A4%EB%A3%A8%EA%B8%B0/2.%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%A0%84%EC%B2%98%EB%A6%AC)
