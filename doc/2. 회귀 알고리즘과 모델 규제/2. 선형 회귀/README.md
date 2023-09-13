# 2-2. 선형 회귀

## k-최근접 이웃 모델의 한계

이전과 같이 모델을 준비한다.

```python
input_train, input_test, target_train, target_test = train_test_split(perch_length, perch_weight, random_state=42)

input_train = input_train.reshape(-1, 1)
input_test = input_test.reshape(-1, 1)

kn = KNeighborsRegressor(n_neighbors=3)
kn.fit(input_train, target_train)
```

이제 이 모델을 이용해서 길이가 50cm인 농어의 무게를 예측한다.

```python
kn.predict([[50]])
# output: [1033.33333333]
```

이 모델은 농어의 무게를 1033kg로 예측했다. 하지만 실제 50cm인 농어의 무게는 이보다 훨씬 더 많이 나간다고 한다.    
문제를 파악하기 위해서 데이터를 시각화 해 본다.

```python
distances, indexes = kn.kneighbors([[50]])

scatter(input_train, target_train)
scatter(input_train[indexes], target_train[indexes], marker="D")
scatter(50, 1033, marker="^")

xlabel("length")
ylabel("weight")

show()
```

![scatter1.png](https://raw.githubusercontent.com/RFLXN/PnP.AI.2023/main/img/2-2/scatter1.png)

위의 이미지를 보면 무게와 길이가 어느정도 비례하게 올라가는것을 알 수 있다.    
하지만 k-최근접 이웃 알고리즘은 주변 이웃의 평균값을 이용해서 예측하기 때문에 새로운 샘플이 훈련 세트의 범위를 벗어나면 값이 이상해 질 수 있다.   
50부터가 이미 범위를 초과했기 때문에, 100, 1000, 10000의 값으로 예측해도 지금과 완전히 같은 결과가 나올 것이다.

```python
distances, indexes = kn.kneighbors([[1000]])

scatter(input_train, target_train)
scatter(input_train[indexes], target_train[indexes], marker="D")
scatter(1000, 1033, marker="^")

xlabel("length")
ylabel("weight")

show()
```

![scatter2.png](https://raw.githubusercontent.com/RFLXN/PnP.AI.2023/main/img/2-2/scatter2.png)

위와같이 길이를 1000으로 줘도 결과가 같은 걸을 알 수 있다.

## 선형 회귀

이럴때 이용할 수 있는 알고리즘이 linear regression 알고리즘이다.    

scikit-learn에서는 linear_model.LinearRegression 클래스를 이용해서 선형 회귀를 사용할 수 있다.

```python
lr = LinearRegression()
lr.fit(input_train, target_train)

lr.predict([[50]])
# output: [1241.83860323]
```

k-최근접 이웃 회귀와 다르게 1241로 값을 더 크게 예측한 것을 알 수 있다.    

선형 회귀는 선형적으로 변화하는 값을 예측하고, 이를 위해 $y = ax + b$ 꼴의 1차 방정식의 $a, b$의 값을 찾는 알고리즘이다.
$a, b$값은 LinearRegression 인스턴스의 coef_와 intercept_ 속성에 저장되어 있다.

```python
print(lr.coef_, lr.intercept_)
# output: [39.01714496] -709.0186449535477
```

이 모델을 시각화하면 다음과 같다.

```python
scatter(input_train, target_train)
plot([15, 50], [15 * lr.coef_ + lr.intercept_, 50 * lr.coef_ + lr.intercept_])  # 학습한 모델의 최적선
scatter(50, 1241.8, marker="^")
xlabel("length")
ylabel("weight")
show()
```

![linear1.png](https://raw.githubusercontent.com/RFLXN/PnP.AI.2023/main/img/2-2/linear1.png)

## 다항 회귀

실제 데이터를 살펴보면 1차 방정식보단 2차 방정식에 가까운 곡선을 보여주고 있다.    
2차 방정식은 $y = ax^2 + bx + c$ 꼴이고, 여기서 x는 무게이므로, 다항 방정식으로 학습시키기 위해서는 $x^2$의 값, 
즉 무게의 제곱의 값이 필요하다.

```python
input_train_poly = column_stack((input_train**2, input_train))
input_test_poly = column_stack((input_test**2, input_test))
```

이렇게 제곱한 값과 원래의 값이 동시에 들어간 데이터를 그대로 fit 해주면 다항으로 학습이 가능하다.

```python
poly_lr = LinearRegression()
poly_lr.fit(input_train_poly, target_train)

poly_lr.predict([[50**2, 50]])
# output: [1573.98423528]
```

이전보다 더 높은 값을 예측한 것을 볼 수 있다.

```python
print(poly_lr.coef_, poly_lr.intercept_)
# output: [  1.01433211 -21.55792498] 116.05021078278259
```

$y = ax^2 + bx + c$ 에서 a는 1.014, b는 -21.557, c는 116.05로 학습되었다.    
이것을 시각화하면

```python
point = arange(15, 50)  # 1~49까지의 배열 생성
scatter(input_train, target_train)
plot(point, 1.01 * point**2 - 21.6 * point + 116.05)    # 학습한 모델의 최적선 (2차 방정식)
scatter(50, 1574, marker="^")
xlabel("length")
ylabel("weight")
show()
```

![linear2.png](https://raw.githubusercontent.com/RFLXN/PnP.AI.2023/main/img/2-2/linear2.png)

단순한 1차 방정식보다 2차 방정식이 훨씬 더 정확한 것을 볼 수 있다.

이런식으로 다항으로 학습한 선형 회귀를 polynomial regression(다항회귀) 라고 한다. 

</br></br></br></br>

[다음 (2-3. 특성 공학과 규제) ->](https://github.com/RFLXN/PnP.AI.2023/tree/main/doc/2.%20%ED%9A%8C%EA%B7%80%20%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98%EA%B3%BC%20%EB%AA%A8%EB%8D%B8%20%EA%B7%9C%EC%A0%9C/3.%20%ED%8A%B9%EC%84%B1%20%EA%B3%B5%ED%95%99%EA%B3%BC%20%EA%B7%9C%EC%A0%9C)
