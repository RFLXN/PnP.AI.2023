# 4-2. 교차 검증과 그리드 서치

## 검증 세트
훈련 세트와 테스트 세트 두개로 나누어서 훈련과 테스트를 진행하다 보면, 과대적합/과소적합을 해결하는 동안
테스트 세트에 대해 잘 맞는 모델이 나오게 될 수 있다.    
이를 방지하기 위해, 테스트 세트를 또 분리하여 검증 세트(validation set)로 이용하여 테스트를 여러번 진행하여 이를 방지할 수 있다.    

하지만, 검증 세트를 만들면 테스트 세트의 크기가 작아져 점수가 불안정해질 수 있다.
이를 방지하기 위하여 교차 검증(cross validation)을 이용한다.    

교차 검증은 교차 세트를 분리하여 점수를 평가하는 과정을 여러번 반복하고, 반복한 점수의 평균을 최종 점수로 이용한다.

scikit-learn에서는 model_selection.cross_validate 메소드를 이용하여 교차 검증을 할 수 있다.

```python
dt = DecisionTreeClassifier(random_state=42)

cross_validate(dt, input_data, target_data)
# output: {'fit_time': array([0.00712395, 0.00585389, 0.00595021, 0.00600982, 0.00613284]), 'score_time': array([0.00060701, 0.00050521, 0.00056982, 0.00049734, 0.00047207]), 'test_score': array([0.83      , 0.78538462, 0.79676674, 0.80754426, 0.83294842])}

mean(score["test_score"])
# output: 0.8105288091431279
```

cross_validata 함수가 리턴한 딕셔너리의 test_score에는 각 교차 검증의 점수가 나오고, 이를 mean 함수로 평균을 내면 교차 검증의 점수라고 볼 수 있다.    

cross_validate 함수는 기본적으로 교차 검증을 하지만, 훈련 세트를 섞어서 폴드를 진행하지는 않는다.
cross_validate 함수의 cv 인수에 분할기 클래스를 지정하고, 분할기 클래스의 shuffle 옵션을 통해서 샘플들을 섞어서 폴드를 진행할 수 있다.
추가적으로 분할기 클래스의 n_splits 옵션을 통해 폴드를 몇번 진행할지도 설정할 수 있다.

```python
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

cross_validate(dt, input_data, target_data, cv=splitter)
# output: {'fit_time': array([0.00664806, 0.00656891, 0.00662208, 0.00659037, 0.00637388,
#       0.00650692, 0.00673318, 0.00645924, 0.00666022, 0.00652385]), 'score_time': array([0.00042486, 0.00037885, 0.00039697, 0.0003829 , 0.00037122,
#       0.00037313, 0.00037575, 0.00036097, 0.00037384, 0.00037313]), 'test_score': array([0.86923077, 0.90461538, 0.88307692, 0.87076923, 0.86153846,
#       0.86461538, 0.89384615, 0.87827427, 0.87827427, 0.87673344])}

mean(score_fold["test_score"])
# output: 0.8780974279957331
```

## 하이퍼 파라미터 튜닝과 그리드 서치

하이퍼 파라미터는 학습 알고리즘에 의해 정해지는것이 아닌, 사람이 직접 지정해야 하는 값이다.    
이 값을 조금씩 튜닝하여 최적의 결과를 얻는 모델을 찾아내야 하는데, 한 모델에 다수의 하이퍼 파라미터가 존재하고, 
서로 다른 파라미터끼리도 영향을 주고받을 수 있기 때문에 일일히 값을 변경해가며 찾는것은 매우 힘들다.    
이를 위해 그리드 서치라는 기법을 이용하고, scikit-learn 에서는 model_selection.GridSearchCV 클래스를 통해서 이용할 수 있다.
이 GridSearchCV 클래스는 내부에서 교차검증도 자동으로 진행하기 때문에 따로 cross_validate() 함수를 이용하지 않아도 된다.    

```python
params = {"min_impurity_decrease": [0.0001, 0.0002, 0.0003, 0.0004]}

# 참고: n_jobs 인자는 그리드 서칭, 학습, 테스트에 사용되는 프로세서 수를 지정한다. -1은 전체 프로세서를 이용한다.
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)

gs.fit(input_train, target_train)

print(gs.best_estimator_)
# output: DecisionTreeClassifier(min_impurity_decrease=0.0003, random_state=42)
```

`min_impurity_decrease` 파라미터에 대해 그리드 서칭을 진행한 결과, 
`min_impurity_decrease=0.0003`에서 최적의 결과가 나왔음을 확인할 수 있다.   

이번에는 여러 파라미터를 이용하여 그리드 서칭을 진행해 보자

```python
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)

gs.fit(input_train, target_train)

print(gs.best_estimator_)
# output: DecisionTreeClassifier(max_depth=15, min_impurity_decrease=0.0001,
#                       min_samples_split=22, random_state=42)

print(max(gs.cv_results_["mean_test_score"]))
# output: 0.8752056020639183
```

`max_depth = 15`, `min_impurity_decrease = 0.0001`, `min_samples_splite = 22`에서 0.8752056020639183의 점수로 최상의 결과가 나온 것을 확인할 수 있다.

## 랜덤 서치
매개변수의 값의 간격을 정하기 어렵거나 할 때는 랜덤 서치를 이용할 수 있다.

랜덤 서치는 위의 GridSearchCV 대신 RandomizedSearchCV 클래스를 이용하고, 사전 지정하는 파라미터를 numpy의 arange 함수 대신,
scipy의 stats 패키지의 uniform, randint 함수를 이용한다.


```python
params = {
    "min_impurity_decrease": uniform(0.0001, 0.001),
    "max_depth": randint(20, 50),
    "min_samples_split": randint(2, 25),
    "min_samples_leaf": randint(1, 25),
}

# n_iter 인수를 통해 샘플링 횟수 지정
rs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1, n_iter=100)

rs.fit(input_train, target_train)

print(rs.best_estimator_)
# output: DecisionTreeClassifier(max_depth=45,
#                       min_impurity_decrease=0.00028442920603391546,
#                       min_samples_leaf=7, min_samples_split=12,
#                       random_state=42)

print(max(gs.cv_results_["mean_test_score"]))
# output: 0.8721263623440215
```
