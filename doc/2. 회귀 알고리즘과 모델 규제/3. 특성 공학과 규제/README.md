# 2-3. 특성 공학과 규제

## 다중 회귀

여러개의 특성을 이용한 선형 회귀를 다중 회귀(multiple regression) 이라고 한다.    
특성이 1개인 경우에는 이전에 본 것처럼 2차원 평면으로 시각화 할 수 있었는데, 2개인 경우에는 3차원 공간으로,
3개부터는 4차원을 표현할 수 없음으로 3차원 공간의 묶음으로 표현할 수 있다.    

이전에는 농어의 길이만 특성으로 이용했지만, 농어의 높이, 두께 등도 특성으로 이용할 수 있다.   
또한, 이전에 존재하던 특성들을 이용해서 새로운 특성을 만들어 낼 수도 있다. (ex: 길이와 높이를 곱한 값을 하나의 특성으로 이용함)    
이렇게 기존의 특성을 이용하여 새로운 특성을 뽑아내는 것을 특성 공학(feature engineering) 이라고 한다.    

## 데이터 준비

우선 판다스를 이용해서 csv 데이터를 불러온다.

```python
from pandas import read_csv

perch_data = read_csv("./data.csv").to_numpy()
```

불러온 csv 데이터와 이전의 무게 데이터를 이용해서 훈련 세트와 테스트 세트를 분리한다.

```python
input_train, input_test, target_train, target_test = train_test_split(perch_data, perch_weight, random_state=42)
```

scikit-learn 라이브러리에는 preprocessing.PolynomialFeatures 클래스를 이용해서 다항 회귀를 위한 특성 공학을 실행할 수 있다.    
이를 이용해서 기존의 데이터로부터 새로운 특성을 뽑아내 보자.

```python
pf = PolynomialFeatures(include_bias=False)     # 변환기 객체 생성
pf.fit(input_train)
input_train_poly = pf.transform(input_train)    # 훈련 세트 특성 변환
input_test_poly = pf.transform(input_test)      # 테스트 세트 특성 변환
```

이렇게 만들어 낸 특성을 이용해서 다중 회귀를 실행해 보자.

```python
lr = LinearRegression()
lr.fit(input_train_poly, target_train)

lr.score(input_train_poly, target_train)
# output: 0.9903183436982124

lr.score(input_test_poly, target_test)
# output: 0.971455991159406
```

단순히 길이만 이용할 때보다 훨씬 더 높은 점수가 나오는 것을 볼 수 있다.    

그렇다면 PolynomialFeatures를 이용해서 더 많은 특성을 만들어 내서 훈련하면 어떻게 될까?    

```python
pf_over = PolynomialFeatures(degree=5, include_bias=False)
pf_over.fit(input_train)
over_input_train_poly = pf_over.transform(input_train)
over_input_test_poly = pf_over.transform(input_test)

print(over_input_train_poly.shape)
# output: (42, 55)
```

degree를 높여주면 디폴트로 특성의 2제곱이 아닌 degree의 값 만큼 제곱하여 특성의 수를 많게 만들 수 있다.   
shape를 출력해 보면 만들어진 특성의 개수가 55개인 것을 확인할 수 있다. 이 특성을 이용해서 훈련해 보면

```python
lr_over = LinearRegression()
lr_over.fit(over_input_train_poly, target_train)

lr_over.score(over_input_train_poly, target_train)
# output: 0.9999999999998099

lr_over.score(over_input_test_poly, target_test)
# output: -144.40606225090627
```

훈련 세트는 매우 높은 점수가 나오지만, 테스트 세트에서 음수가 나올 정도로 낮은 점수가 나온다.    
특성의 개수를 늘리면 선형 모델은 매우 강력해지지만, 그만큼 과적합 되기 쉽다.    

## 규제
규제(regularization)는 모델이 훈련 세트를 너무 과도하게 학습하지 못하도록 막는 것을 말한다.    
선형 회귀 모델에서는 특성의 계수를 줄여서 모델의 규제를 할 수 있다.    

scikit-learn에서는 preprocessing.StandardScaler 클래스를 이용해서 특성을 표준점수화 할 수 있다.

```python
ss = StandardScaler()
ss.fit(over_input_train_poly)

input_train_scaled = ss.transform(over_input_train_poly)
input_test_scaled = ss.transform(over_input_test_poly)
```

선형 회귀에 규제를 추가한 모델을 릿지(ridge) 회귀와 라쏘(lasso) 회귀라고 한다.
릿지 회귀는 계수를 제곱한 값을 기준으로 규제를 적용하고, 라쏘 회귀는 계수의 절댓값을 기준으로 규제를 적용한다.

## 릿지 회귀
scikit-learn.linear 패키지 내부에는 LinearRegression 이외에도 Ridge 클래스가 있어, 이를 이용해 릿지 회귀를 사용할 수 있다.    

```python
ridge = Ridge()
ridge.fit(input_train_scaled, target_train)

ridge_train_score = ridge.score(input_train_scaled, target_train)
# output: 0.9896101671037343

ridge_test_score = ridge.score(input_test_scaled, target_test)
# output: 0.9790693977615391
```

55개의 특성을 이용했음에도 불구하고, 과대적되지 않고 정상적인 점수가 나오는 것을 확인할 수 있다.    

릿지와 라쏘 모델에서는 alpha값을 변경하여 규제가 적용되는 정도를 변경할 수 있다.    
알파값이 높으면 규제의 강도가 세지고, 알파값이 낮으면 규제가 약해져서 과대적합 될 가능성이 높아진다.    

적절한 알파 값을 찾는 방법 중 하나는, 알파값에 대한 $R^2$ 그래프를 그려 보는 것이다.     
훈련 세트와 테스트 세트의 점수가 가장 가까운 지점이 최적의 알파값이 된다.

```python
train_scores = []
test_scores = []

alpha_values = [0.001, 0.01, 1, 10, 100]

for alpha in alpha_values:
    current_ridge = Ridge(alpha=alpha)
    current_ridge.fit(input_train_scaled, target_train)
    train_scores.append(current_ridge.score(input_train_scaled, target_train))
    test_scores.append(current_ridge.score(input_test_scaled, target_test))

plot(log10(alpha_values), train_scores)
plot(log10(alpha_values), test_scores)
xlabel("alpha")
ylabel("R^2")
show()
```

![alpha1.png](https://raw.githubusercontent.com/RFLXN/PnP.AI.2023/main/img/2-3/alpha1.png)

위의 파란색 선이 훈련 세트, 아래의 주황색 선이 테스트 세트이다.    
보면 alpha = -1, 즉 $10^-1$인 0.1에서 가장 점수차가 적은 것을 볼 수 있다.    

이를 이용해 알파값을 0.1로 적용시켜 보면

```python
alpha_ridge = Ridge(alpha=0.1)
alpha_ridge.fit(input_train_scaled, target_train)

alpha_ridge.score(input_train_scaled, target_train)
# output: 0.9903815817570367

alpha_ridge.score(input_test_scaled, target_test)
# output: 0.9827976465386918
```

훈련 세트와 테스트 세트의 점수가 비슷하여 과대적합과 과소적합 사이에서 균형을 이루는 것을 알 수 있다.

## 라쏘 회귀
Ridge와 마찬가지로 Lasso 클래스를 이용하면 라쏘 회귀를 이용할 수 있다.

```python
lasso = Lasso()
lasso.fit(input_train_scaled, target_train)

lasso_train_score = lasso.score(input_train_scaled, target_train)
# output: 0.989789897208096

lasso_test_score = lasso.score(input_test_scaled, target_test)
# output: 0.9800593698421884
```

라쏘에서도 최적의 알파 값을 찾아 보면,

```python
lasso_train_scores = []
lasso_test_scores = []

for alpha in alpha_values:
    current_lasso = Lasso(alpha=alpha)
    current_lasso.fit(input_train_scaled, target_train)
    lasso_train_scores.append(current_lasso.score(input_train_scaled, target_train))
    lasso_test_scores.append(current_lasso.score(input_test_scaled, target_test))
```

![alpha2.png](https://raw.githubusercontent.com/RFLXN/PnP.AI.2023/main/img/2-3/alpha2.png)

이번에는 0.1이 아니라 10이 최적의 알파값임을 알 수 있다.    

alpha=10을 적용시켜 보면, 

```python
alpha_lasso = Lasso(alpha=10)
alpha_lasso.fit(input_train_scaled, target_train)

alpha_lasso.score(input_train_scaled, target_train)
# output: 0.9888067471131866

alpha_lasso.score(input_test_scaled, target_test)
# output: 0.9824470598706695
```

이번에도 과대적합도 과소적합도 아닌 적절하게 학습되었음을 알 수 있다.

</br></br></br></br>

[다음 (3-1. 로지스틱 회귀) ->](https://github.com/RFLXN/PnP.AI.2023/tree/main/doc/3.%20%EB%8B%A4%EC%96%91%ED%95%9C%20%EB%B6%84%EB%A5%98%20%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98/1.%20%EB%A1%9C%EC%A7%80%EC%8A%A4%ED%8B%B1%20%ED%9A%8C%EA%B7%80)

