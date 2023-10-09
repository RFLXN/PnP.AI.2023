# 4-3. 트리의 앙상블

* 앙상블 학습(ensemble learning): 정형 데이터에서 대체로 가장 뛰어난 성과를 내는 알고리즘

## 랜덤 포레스트

* 앙상블 학습의 가장 대표적인 방법 중 하나
* 부트스트랩 샘플: 훈련 데이터에서 랜덤하게 추출하는 샘플. 추출된 샘플은 중복될 수 있다.
* 랜덤 포레스트를 부트스트래핑한 여러 데이터세트들을 결정 트리로 훈련시키고,
트리 훈련 과정에서 노드를 분리할 때 특성을 랜덤으로 골라 최선의 분할을 찾는다.
* scikit-learn에서는 ensemble.RandomForestClassifier를 통해 이용할 수 있다.    


RandomForestClassifier를 실제로 사용해 보자.

```python
rf = RandomForestClassifier(n_jobs=-1, random_state=42)

scores = cross_validate(rf, input_train, target_train, return_train_score=True, n_jobs=-1)

print(mean(scores["train_score"]))
# output: 0.997844759088341

print(mean(scores["test_score"]))
# output: 0.8914208392565683
```

과대적합된 결과가 나왔음을 알 수 있다.    


OOB (Out Of Bag): 부트스트래핑에서 포함되지 않은 나머지 샘플들을 뜻한다.    

랜덤 포레스트 모델의 oob_score_ 프로퍼티를 통해서 OOB 모델로 테스트를 진행한 결과를 볼 수 있으며,
이때는 RandomForestClassifier 클래스의 생성자 인수에 oob_score=True로 넣어줘 이용할 수 있다.

```python
rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)
rf.fit(input_train, target_train)

print(rf.oob_score_)
# output: 0.8981937602627258
```

## 엑스트라 트리

랜덤 포레스트와 매우 비슷하지만, 학습 세트를 부트스트래핑 하지 않고 전체 샘플을 이용한다.
또한, 노드를 분할할 때 최적의 분할을 찾지 않고 랜덤으로 분할시킨다.

ensemble.ExtraTreeClassifier 클래스를 통해 이용 가능하다.

```python
et = ExtraTreesClassifier(n_jobs=-1, random_state=42)

scores = cross_validate(et, input_train, target_train, return_train_score=True, n_jobs=-1)

print(mean(scores["train_score"]))
# output: 0.997844759088341

print(mean(scores["test_score"]))
# output: 0.8903937240035804
```

## 그래디언트 부스팅

* 그래디언트 부스팅(gradient boosting)은 깊이가 얕은 결정 트리를 이용하여 이전 트리의 오차를 보완하는 방식이다.    
* 깊이가 얕은 트리를 이용하기 때문에 과대적합에 강하다. (일반화 성능이 높다)
* 경사 하강법을 이용한다.

ensemble.GradientBoostingClassifier 클래스를 통해 이용 가능하다.

```python
gb = GradientBoostingClassifier(random_state=42)

scores = cross_validate(gb, input_train, target_train, return_train_score=True, n_jobs=-1)

print(mean(scores["train_score"]))
# output: 0.8894704231708938

print(mean(scores["test_score"]))
# output: 0.8715107671247301
```

실제로 사용해 보면 위의 두 트리 앙상블에 비해 과대적합되지 않음을 볼 수 있다.

## 히스토그램 기반 그래디언트 부스팅
* 히스토그램 기반 그래디언트 부스팅(histogram-based gradient boosting)은 정형 데이터를 다루는 머신러닝 알고리즘 중 가장 인기가 높은 알고리즘이다.
* 입력 특성을 256개의 고정된 구간으로 나누어 최적 분할을 빠르게 찾을 수 있다.
* 256개의 구간 중 하나를 누락된 값을 위해서 사용한다. -> 누락된 특성을 위해 전처리 할 필요가 없다.

ensemble.HistGradientBoostingClassifier를 통해 이용할 수 있다.

```python
hg = HistGradientBoostingClassifier(random_state=42)
scores = cross_validate(hg, input_train, target_train, return_train_score=True, n_jobs=-1)

print(mean(scores["train_score"]))
# output: 0.9380129799494501

print(mean(scores["test_score"]))
# output: 0.8805410414363187
```

## 다른 라이브러리

그래디언트 부스팅을 구현한 파이썬 라이브러리는 scikit-learn 이외에도 여러 종류가 존재한다.    
MS의 LightGMB이나 XGBoost등이 대표적이다.
