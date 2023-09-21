# 3-2. 확률적 경사 하강법

## 점진적 학습

머신러닝에서 새로운 데이터(샘플)가 지속적으로 추가될 때, 이전의 모델을 재훈련하지 않고,
새로 추가된 데이터에 대해서만 추가적으로 학습시키는 것을 점진적 학습이라고 부른다.    
점진적 학습 방법 중 가장 대표적인 것이 확률적 경사 하강법(Stochastic Gradient Descent) 이다.

## 확률적 경사 하강법

* 확률적 경사 하강법: 랜덤한 훈련 세트 하나를 골라 가장 가파른 길을 찾고, 그 경사를 따라 내려오는 학습 방법
* 미니배치(minibatch) 경사 하강법: 랜덤한 훈련 세트 여러개를 골라 확률적 경사 하강법과 동일하게 진행함
* 배치(batch) 경사 하강법: 훈련 세트 전체를 이용하여 확률적 경사 하강법과 동일하게 진행함
* 에포크(epoch): 경사 하강법에서 훈련 세트를 모두 사용했음에도 경사를 모두 내려오지 못했을 때, 다시 훈련 세트를 모두 채워 넣고 하강을 시작한다.
    여기서 훈련 세트를 모두 사용하는 한 번의 사이클을 에포크라고 한다. 보통 경사 하강법은 수백, 수천번의 에포크를 진행한다.

## 손실 함수

손실 함수(loss function)는 어떤 문제에서 해당 모델이 얼마나 엉터리인지를 측정하는 기준이다. 즉, 값이 작을수록 좋다는 뜻이다.    
하지만 해당 모델의 손실 함수 최소값을 알지 못하기에, 만족할 만한 수준이 나올때까지 학습을 여러번 진행해야 한다.    

## 로지스틱 손실 함수

양성 클래스 (정답일 때)는 $-log{(예측확률)}$, 음성 클래스일때는 $-log{(1-{예측확률})}$로 손실을 정의 하면,
양성 클래스에서는 확률이 1에서 멀어질수록 값이 매우 큰 양수가 되고, 음성 클래스에서는 확률이 0에서 멀어질수록 값이 매우 큰 양수가 된다.   
이러한 함수를 로지스틱 손실 함수, 또는 이진 크로스엔트로피 손실 함수(binary cross-entropy loss function)라고 부른다.

## 확률적 경사 하강법의 실행

scikit-learn 에서는 linear_model.SGDClassifier 클래스를 통해서 확률적 경사 하강법을 사용할 수 있다.    

우선은 이전과 같이 데이터를 준비해 주고,
```python
input_data = fish_data[["Weight", "Length", "Diagonal", "Height", "Width"]].to_numpy()
target_data = fish_data["Species"].to_numpy()

input_train, input_test, target_train, target_test = train_test_split(input_data, target_data, random_state=42)

ss = StandardScaler()
ss.fit(input_train, target_train)

input_train_scaled = ss.transform(input_train)
input_test_scaled = ss.transform(input_test)
```

이후 SGDClassifier를 이용해서 학습을 진행한다.

> 참고: 책에는 loss="log"로 나와있지만, log_loss가 로지스틱 손실 함수임.
> ```
> raise InvalidParameterError(
> sklearn.utils._param_validation.InvalidParameterError: The 'loss' parameter of SGDClassifier must be a str among {'squared_error', 'log_loss', 'modified_huber', 'squared_hinge', 'epsilon_insensitive', 'huber', 'perceptron', 'hinge', 'squared_epsilon_insensitive'}. Got 'log' instead.
> ```
```python
# loss="log": 손실 함수를 로지스틱 손실 함수로 지정
# max_iter=10: 에포크 최대 횟수
sc = SGDClassifier(loss="log_loss", max_iter=10, random_state=42)
sc.fit(input_train_scaled, target_train)

sc.score(input_train_scaled, target_train)
# output: 0.773109243697479

sc.score(input_test_scaled, target_test)
# output: 0.775
```

또한, 경사 하강법이므로, 추가적으로 학습을 진행할 수 있다.
```python
sc.partial_fit(input_train_scaled, target_train)

sc.score(input_train_scaled, target_train)
# output: 0.8151260504201681

sc.score(input_test_scaled, target_test)
# output: 0.85
```

추가적으로 학습을 진행하니 점수가 높아지는 것을 알 수 있다.

## 에포크와 과대/과소적합

에포크 횟수가 적으면 그만큼 훈련의 횟수가 적다는 뜻이므로 과소적합이 되고,
에포크 횟수가 많으면 반대로 과대적합이 될 수 있다.    

에포크가 진행됨에 따라 정확도가 꾸준히 증가하지만, 어느 순간 부터는 오히려 감소하기 시작한다.    
이 지점이 모델이 과대적합되기 시작하는 부분이다.    
그러니 좋은 모델을 만들기 위해서는 모델이 과대적합을 시작하기 전에 학습을 (에포크를) 멈춰야 하는데, 이를 조기 종료(early stopping)라고 한다.　　　　

에포크에 따른 정확도를 시각화 해 보면
```python
sc_partial = SGDClassifier(loss="log_loss", max_iter=10, random_state=42)
partial_train_score = []
partial_test_score = []
classes = unique(target_train)

for _ in range(0, 300):
    sc_partial.partial_fit(input_train_scaled, target_train, classes=classes)
    partial_train_score.append(sc_partial.score(input_train_scaled, target_train))
    partial_test_score.append(sc_partial.score(input_test_scaled, target_test))

plot(partial_train_score)
plot(partial_test_score)
xlabel("epoch")
ylabel("score")
show()
```

![epoch1.png](https://raw.githubusercontent.com/RFLXN/PnP.AI.2023/main/img/3-2/epoch1.png)

약 100번째 에포크 이후로부터 점수차가 조금씩 벌어지면서 점점 더 과대적합 되는것을 확인할 수 있다.   
따라서 에포크를 100으로 맞추고 점수를 출력 해 보면,

```python
# tol=None: SGDClassifier는 자체적으로 에포크 횟수가 일정 이상이 지났는데도
# 점수가 올라가지 않으면 학습을 종료하는데, tol=None이면 이를 체크하지 않는다.
sc_100 = SGDClassifier(loss="log_loss", max_iter=100, random_state=42, tol=None)
sc_100.fit(input_train_scaled, target_train)

sc_100.score(input_train_scaled, target_train)
# output: 0.957983193277311

sc_100.score(input_test_scaled, target_test)
# output: 0.925
```

더 좋은 점수가 나오는 것을 확인할 수 있다.


</br></br></br></br>

[다음 (4-1. 결정 트리) ->](https://github.com/RFLXN/PnP.AI.2023/tree/main/doc/4.%20%ED%8A%B8%EB%A6%AC%20%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98/1.%20%EA%B2%B0%EC%A0%95%20%ED%8A%B8%EB%A6%AC)
