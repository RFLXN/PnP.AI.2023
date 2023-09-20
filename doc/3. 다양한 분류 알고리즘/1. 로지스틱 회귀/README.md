# 3-1. 로지스틱 회귀

## 확률 구하기: k-최근접 이웃 모델

특정한 샘플의 특정한 샘플 값들을 통해서 해당 샘플이 무슨 클래스일지의 확률을 구한다고 하자.    
이를 k-최근접 이웃 모델로는 해당 샘플의 이웃의 비율을 통해서 확률을 유추할 수 있을것이다.

다음과 같은 형식의 csv 데이터가 있다.
```csv
Species,Weight,Length,Diagonal,Height,Width
Bream,242,25.4,30,11.52,4.02
Bream,290,26.3,31.2,12.48,4.3056
Bream,340,26.5,31.1,12.3778,4.6961
```

위의 데이터를 특성(input)과 클래스(target)으로 나눠보면    
Species가 타겟이고 나머지 데이터를 특성이라고 볼 수 있다.    

이를 훈련에 필요한 데이터로 가공하면 다음과 같다.

```python
fish_data = read_csv("./data.csv")

# 타겟과 인풋 데이터 분리
input_data = fish_data[["Weight", "Length", "Diagonal", "Height", "Width"]].to_numpy()
target_data = fish_data["Species"].to_numpy()

# 훈련 세트와 테스트 세트 분리
input_train, target_train, input_test, target_test = train_test_split(input_data, target_data, random_state=42)

# 인풋 데이터 표준화 전처리
ss = StandardScaler()
ss.fit(input_train, target_train)

input_train_scaled = ss.transform(input_train)
input_test_scaled = ss.transform(input_test)
```

이를 훈련하고 테스트 해 보면

```python
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(input_train_scaled, target_train)

kn.score(input_train_scaled, target_train)
# output: 0.8907563025210085

kn.score(input_test_scaled, target_test)
# output: 0.85
```

이런식으로 타겟 클래스가 2개 이상인 분류를 다중 분류(multi-class classification)라고 한다.    

사이킷런의 분류 모델로 다중 분류를 예측할 때, predict() 메소드는 해당 샘플의 클래스명(여기서는 Bream 등)을 반환해 준다.

```python
kn.predict(input_test_scaled[:5])
# output: ['Perch' 'Smelt' 'Pike' 'Perch' 'Perch']
```

predict_proba() 메소드는 각 클래스일 확률을 반환해 준다.

```python
kn.predict_proba(input_test_scaled[:5])
# output: [[0.         0.         1.         0.         0.         0.         0.        ]
#         [0.         0.         0.         0.         0.         1.         0.        ]
#         [0.         0.         0.         1.         0.         0.         0.        ]
#         [0.         0.         0.66666667 0.         0.33333333 0.         0.        ]
#         [0.         0.         0.66666667 0.         0.33333333 0.         0.        ]]
```

여기서 각 확률의 클래스 순서는 classes_ 의 클래스 순서와 같다. 

```python
print(kn.classes_)
# output: ['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']
```

즉, 0번째는 Bream일 확률, 2번째는 Parkki일 확률, ..., 6번째는 Whitefish일 확률이 된다.   

실제로 저 확률이 맞는지 이웃을 확인해 보면

```python
# 3번째 샘플의 이웃
# 즉, [0.         0.         0.66666667 0.         0.33333333 0.         0.        ] 의 확률을 가진 샘플
kn_distances, kn_indexes = kn.kneighbors(input_test_scaled[3:4])
print(target_train[kn_indexes])
# output: [['Roach' 'Perch' 'Perch']]
```

Roach 1개, Perch 2개로 `['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']`의 2번째 클래스가 2개,
4번째 클래스가 1개로 확률이 정확한 것을 알 수 있다.    

하지만 k-최근접 이웃 모델로는, 확률이 무조건 이웃의 개수에 따라서 정해지고, 이웃의 개수가 적으면 확률의 범위가 넓고 부정확해진다는 단점이 있다.

## 로지스틱 회귀

logistic regression은 regression 이지만 분류 모델이다.    
로지스틱 회귀 모델은 선형 회귀와 동일하게 선형 방정식을 학습한다.    
이는 다중 회귀를 위한 선형 방정식과 비슷/동일하다.   
하지만 다중 회귀의 결과는 확률을 리턴하지 않고 심지어는 음수까지도 나오는데, 이를 확률로(0~1 혹은 0%~100%) 변환하는 과정이 필요하다.    
이 때 이용하는 것이 시그모이드 함수(sigmoid function) 혹은 로지스틱 함수(logistic function)라고 부르는 함수이다.    

시그모이드 함수는 다음과 같다.
$$\phi = {1 \over 1 + e^-z}$$
이 함수는　$z$의 값에 따라 음수의 무한대가 0에 수렴하고, 양수의 무한대가 1에 수렴하도록 하는 함수이다.    

scikit-learn 에서는 linear_model.LogisticRegression 클래스를 이용해서 로지스틱 회귀를 할 수 있다.

```python
# C = 규제를 적용하는 정도 (값이 높을수록 규제 정도가 약해짐) / max_iter = 모델 훈련에 사용되는 반복의 최대 개수
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(input_train_scaled, target_train)

lr.score(input_train_scaled, target_train)
# output: 0.9327731092436975

lr.score(input_test_scaled, target_test)
# output: 0.925
```

점수는 충분해 보이니 실제 예측한 결과를 살펴보면

```python
print(lr.predict(input_test_scaled[:5]))
# output: ['Perch' 'Smelt' 'Pike' 'Roach' 'Perch']

lr_proba = lr.predict_proba(input_test_scaled[:5])
print(round(lr_proba, decimals=3))  # 소숫점 3자리 반올림
# output: [[0.    0.014 0.841 0.    0.136 0.007 0.003]
#         [0.    0.003 0.044 0.    0.007 0.946 0.   ]
#         [0.    0.    0.034 0.935 0.015 0.016 0.   ]
#         [0.011 0.034 0.306 0.007 0.567 0.    0.076]
#         [0.    0.    0.904 0.002 0.089 0.002 0.001]]

print(lr.classes_)
# output: ['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']
```

첫 5개가 각각 Perch, Smelt, Pick, Roach, Perch로 예측되었고, 
classes_의 순서와 predict_proba()의 확률을 보면
0번째는 2번째 클래스(Perch), 1번째는 5번째 클래스(Smelt), ..., 4번째는 다시 2번째 클래스로 정확하다는것을 확인할 수 있다.    

로지스틱 회귀 클래스도 선형회귀와 같이 coef_ 값과 intercept_ 값을 확인할 수 있다.

```python
print(lr.coef_.shape, lr.intercept_.shape)
# output: (7, 5) (7,)
```

coef_ 값은 특성의 개수 5개와 같이 열의 개수가 5인것을 알 수 있다.
반면에 coef_의 행의 개수와과 intercept_의 길이가 7개, 즉 클래스(생선 종류)의 개수와 같다는 것을 볼 수 있는데,    
이를 통해서 로지스틱 회귀의 다중 분류는 각 클래스에 대한 확률을 모두 따로 계산한다는 것을 알 수있다.

로지스틱 회귀의 다중 분류는, 각 클래스에 대한 선형 회귀와 같은 다항방정식의 값을 계산한 뒤에,
소프트맥스(softmax) 함수를 통해서 7개의 값을 확률로 변환한다.    
소프트맥스 함수의 계산은 다음과 같다.

$$sum = {e^z1} + {e^z2} + {e^z3} + {e^z4} + {e^z5} + {e^z6} + {e^z7}$$    

$${s1 = {{e^z1} \over sum}}, {s2 = {{e^z2} \over sum}}, ..., {s7 = {{e^z7} \over sum}}$$

각 $z$는 각 클래스에 대한 다항식의 계산 결과, 각 $s$는 그 클래스에 대한 확률이다.

</br></br></br></br>

[다음 (3-2. 확률적 경사 하강법) ->](https://github.com/RFLXN/PnP.AI.2023/tree/main/doc/3.%20%EB%8B%A4%EC%96%91%ED%95%9C%20%EB%B6%84%EB%A5%98%20%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98/2.%20%ED%99%95%EB%A5%A0%EC%A0%81%20%EA%B2%BD%EC%82%AC%20%ED%95%98%EA%B0%95%EB%B2%95)


