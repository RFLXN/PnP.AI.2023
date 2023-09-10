# 1-2. 데이터 전처리

## numpy.column_stack(tup)
(From official document)
>    Stack 1-D arrays as columns into a 2-D array.    
    Take a sequence of 1-D arrays and stack them as columns
    to make a single 2-D array. 2-D arrays are stacked as-is,
    just like with `hstack`.  1-D arrays are turned into 2-D columns
    first.    
    Parameters -> 
    tup : sequence of 1-D or 2-D arrays.    
        Arrays to stack. All of them must have the same first dimension.    
    Returns -> 
    stacked : 2-D array    
        The array formed by stacking the given arrays.

2차원 tuple/list를 인자로 주면, 두 리스트를 같은 인덱스끼리 묶어서 기존 2차원과 축이 바뀐 2차원 리스트를 반환한다.

```python
# example

numpy.column_stack(([1, 2, 3], [4, 5, 6]))
# output: array([1, 4], [2, 5], [3, 6])
```

## train_test_split(*arrays)

Optional arguments

* test_size: float or int. float인경우 0-1사이로 %로 계산. 테스트 세트의 크기.
* train_size: 위와 같지만, 훈련 세트의 크기.
* random_state: random seed.
* shuffle: boolean. random_state를 기반으로 데이터셋들을 섞을지 유무. False인경우 random하게 섞지 않고 배열을 그대로 길이만큼 나눔.
* stratify: array-like. target 데이터를 넣어주면 해당 타겟의 클래스 비율에 따라서 테스트/훈련 세트를 나눔.

scikit-learn 라이브러리의 테스트 세트와 훈련 세트를 자동으로 섞어서 나눠주는 함수.

```python
# example

input_train, input_test, target_train, target_test = sklearn.model_selection.train_test_split(
    fish_data, fish_target, random_state=42
)
```

## 이상한 값

우선 위의 함수들을 이용해 데이터를 준비하고, 모델을 만든다.
```python
# 데이터 가공
fish_input = numpy.column_stack((fish_length, fish_weight))
fish_target = numpy.concatenate((numpy.ones(35), numpy.zeros(14)))

# 데이터를 훈련/테스트 세트로 분할
input_train, input_test, target_train, target_test = train_test_split(
    fish_input, fish_target, random_state=42, stratify=fish_target
)

# 모델 준비
kn = KNeighborsClassifier()
kn.fit(input_train, target_train)
```

이곳에 길이 25, 무게 150의 데이터를 예측시켜 본다.
```python
kn.predict([[25, 150]])
# output: [0.]
```

길이가 25, 무게가 150인데 도미(1)이 아닌 빙어(0)으로 예측한다.    

이는 잘못된 예측이므로, 시각화하여 데이터를 표시해 본다.

```python
scatter(input_train[:, 0], input_train[:, 1])
scatter(25, 150, marker="^")
xlabel("length")
ylabel("weight")
show()
```

![scatter1.png](https://raw.githubusercontent.com/RFLXN/PnP.AI.2023/main/img/1-2/scatter1.png)

삼각형으로 표시되는 (25, 150)의 데이터는 분명히 오른쪽 위의 도미쪽에 가까워 보인다.    

하지만, 이 (25, 150)의 가장 가까운 이웃을 표시하면 다음과 같다.

```python
# KNeighborsClassifier 클래스의 kneighbors 메소드를 이용하면 최근접 이웃을 가져올 수 있다.
distances, indexes = kn.kneighbors([[25, 150]])

scatter(input_train[:, 0], input_train[:, 1])
scatter(25, 150, marker="^")
scatter(input_train[indexes, 0], input_train[indexes, 1], marker="D")
xlabel("length")
ylabel("weight")
show()
```

![scatter2.png](https://raw.githubusercontent.com/RFLXN/PnP.AI.2023/main/img/1-2/scatter2.png)

삼각형의 가장 가까운 이웃 5개(KNeighborsClassifier의 n_neighbors의 디폴트 값 = 5)가 위와 같이 초록색으로 표시되는것을 볼 수 있다.    
5개의 이웃들 중 4개의 이웃이 도미가 아닌 빙어쪽에 분포해 있다.

k-최근접 이웃은 주어진 값(예측할 샘플)의 이웃들(이웃의 개수는 n_neighbors로 지정) 중, 더 많은 클래스(여기서는 빙어=4, 도미=1 이므로 빙어)로 에측하기 때문에 
상식적으로는 도미라고 해도 k-최근접 이웃 모델로는 빙어로 추론 된 것이다.

## 데이터 전처리
이번에는 이웃끼리의 거리를 비교해 보자.
```python
print(input_train[indexes])
# output: [[[ 25.4 242. ]
#  [ 15.   19.9]
#  [ 14.3  19.7]
#  [ 13.   12.2]
#  [ 12.2  12.2]]]

print(distances)
# output: [[ 92.00086956 130.48375378 130.73859415 138.32150953 138.39320793]]
```

이전의 산점도를 보면, 값이 (25.4, 242)인 샘플 (가장 가까운 도미쪽 샘플)과의 거리가 92인데 나머지 빙어쪽 샘플과의 거리가 130으로
도미쪽 샘플과 빙어쪽 샘플과의 거리차가 생각보다 크지 않다는것을 볼 수 있다.    

이는, x축(길이)는 눈금의 범위가 10~40인데 y축(무게)는 0~1000으로 범위가 훨씬 넓기 때문이다.    
이러한 것을 두 특성의 scale이 다르다고 말한다.    

이렇게 특성 간의 기준이 다르면 알고리즘이 올바르게 학습/예측하기 힘들어진다.    
이런것들을 보정하는 작업을 data preprocessing(데이터 전처리) 라고 한다.    

가장 일반적으로 사용되는 데이터 전처리 방법은 standard score(표준 점수)이다.    
표준점수는 각 특성의 값이 평균에서 표준편차의 몇 배만큼 떨어져 있는지를 나타내며,
계산은 평균을 빼고 표준편차를 나누면 된다.

```python
mean = numpy.mean(input_train, axis=0)  # 평균
std = numpy.std(input_train, axis=0)    # 표준편차
scaled_input_train = input_train - mean / std   # 표준점수
```

이제 이 표준점수를 (25, 150)의 샘플에도 적용시켜 준다.
```python
new = ([25, 150] - mean) / std
```

이 표준점수를 적용한 샘플들을 시각화하면 다음과 같다.

```python
scatter(scaled_input_train[:, 0], scaled_input_train[:, 1])
scatter(new[0], new[1], marker="^")
xlabel("length")
ylabel("weight")
show()
```
![scatter3.png](https://raw.githubusercontent.com/RFLXN/PnP.AI.2023/main/img/1-2/scatter3.png)

이전과 크게 달라진 것이 없어 보이지만, length/weight의 범위가 -1.5~1.5로 동일하게 설정 된 것을 볼 수 있다.    

이 표준점수를 적용한 데이터로 다시 머신러닝을 진행해 보자.

```python
# k-최근접 이웃 모델 클래스 생성과 학습
st_kn = KNeighborsClassifier()
st_kn.fit(scaled_input_train, target_train)

# 테스트 데이터도 표준점수 적용
scaled_input_test = (input_test - mean) / std

st_kn.score(scaled_input_test, target_test)
# output: 1.0
```

(25, 150)의 데이터를 예측시켜 보면
```python
st_kn.predict([new])
# output: [1.]
```

정상적으로 예측하는것을 볼 수 있다.   

표준점수를 적용한 샘플들에서 (25, 150)의 이웃들을 시각화 해 보면
```python
st_distances, st_indexs = st_kn.kneighbors([new])

scatter(scaled_input_train[:, 0], scaled_input_train[:, 1])
scatter(new[0], new[1], marker="^")
scatter(scaled_input_train[st_indexs, 0], scaled_input_train[st_indexs, 1], marker="D")
xlabel("length")
ylabel("weight")
show()
```
![scatter4.png](https://raw.githubusercontent.com/RFLXN/PnP.AI.2023/main/img/1-2/scatter4.png)

빙어가 아닌 도미쪽에 이웃이 5개 모두 존재함을 알 수 있다.

</br></br></br></br>

[다음 (2-1. k-최근접 이웃 회귀) ->](https://github.com/RFLXN/PnP.AI.2023/tree/main/doc/2.%20%ED%9A%8C%EA%B7%80%20%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98%EA%B3%BC%20%EB%AA%A8%EB%8D%B8%20%EA%B7%9C%EC%A0%9C/1.%20k-%EC%B5%9C%EA%B7%BC%EC%A0%91%20%EC%9D%B4%EC%9B%83%20%ED%9A%8C%EA%B7%80)