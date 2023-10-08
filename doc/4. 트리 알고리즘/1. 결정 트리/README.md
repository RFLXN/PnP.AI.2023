# 4-1. 결정 트리

다음과 같은 형식의 csv 데이터가 있다
```csv
alcohol,sugar,pH,class
9.4,1.9,3.51,0.0
9.8,2.6,3.2,0.0
9.8,2.3,3.26,0.0
9.8,1.9,3.16,0.0
```

이를 결정 트리 모델을 이용하여 학습시켜 보자.
scikit-learn에서는 tree.DecisionTreeClassifier 클래스를 통해서 이용할 수 있다.

```python
data = wine_data[["alcohol", "sugar", "pH"]].to_numpy()
target = wine_data["class"].to_numpy()

# 학습/테스트 데이터 분할
input_train, input_test, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state=42)

# 학습
# 참고: 결정 트리 모델에서는 StandardScaler를 통한 전처리가 필요하지 않다.
dt = DecisionTreeClassifier(random_state=42)
dt.fit(input_train, target_train)

dt.score(input_train, target_train)
# output: 0.996921300750433

dt.score(input_test, target_test)
# output: 0.8592307692307692
```

훈련 세트에서의 점수가 매우 높은 것으로 보아 과대적합이라고 볼 수 있다.    

결정 트리는 말 그대로 tree의 형태이므로 시각화할 수 있다.


```python
figure(figsize=(10, 7))     # in matplotlib.pyplot
plot_tree(dt)       # in sklearn.tree
plt.show()
```

![tree1.png](https://raw.githubusercontent.com/RFLXN/PnP.AI.2023/main/img/4-1/tree1.png)

위의 이미지를 보면 depth가 너무 깊어 알아보기 어려운것을 알 수 있다.
이를 간략화 해 보자.

```python
figure(figsize=(10, 7))
plot_tree(dt, max_depth=1, filled=True, feature_names=["alcohol", "sugar", "pH"])
show()
```

`plot_tree` 함수에서 max_depth를 통해서 최대 depth를 설정 가능하고, feature_names　인자를 통해 특성의 이름을 설정할 수 있다.

![tree2.png](https://raw.githubusercontent.com/RFLXN/PnP.AI.2023/main/img/4-1/tree2.png)

위의 트리의 구성은 다음과 같다.

1. 테스트 조건 (sugar, alcohol, pH의 값의 조건)
2. 불순도
3. 현재 노드에 속하는 총 샘플 수
4. 클래스별 샘플 수
5. 테스트 조건을 만족하는 샘플은 왼쪽, 만족하지 못하면 오른쪽 노드로 이동

결정 트리는 리프 노드 (최하단 노드)에서 가장 많은 클래스가 예측 클래스로 정해진다.    

결정 트리에서 불순도는 gini impurity(지니 불순도)를 의미하며, 이 지니 불순도 값은 노드에서 데이터를 분할하는 기준을 정하는 데에 사용된다.    
지니 불순도는 다음곽 같은 식으로 나타난다.

$${지니 불순도} = 1 - ({{음성 클래스 비율}^2} + {{양성 클래스 비율}^2})$$

분류가 아닌 회귀면 클래스가 더 많아질 수도 있지만, 기본적으로는 동일한 방식의 수식을 이용한다.    

결정 트리에서는 부모 노드와 자식 노드간의 불순도 차이가 가능한 커지도록 트리를 구성한다.

$${불순도 차이} = {부모의 불순도} - ({왼쪽노드 샘플수}/{부모의 샘플수}) × {왼쪽노드 불순도} - ({오른쪽 노드 샘플 수} / {부모의 샘플 수}) × {오른쪽 노드 불순도}$$

이러한 불순도 차이를 정보 이득(information gain)이라고 부른다.

scikit-learn의 DecisionTree 클래스들에서는 클래스 생성자 인수에 criterion="불순도 종류"를 지정하여 지니 불순도 이외의 불순도도 사용할 수 있고, 
대표적으로는 entropy 불순도가 있다.

이번에는 과대적합된 모델을 해결해 보자.

결정 트리는 가지치기를 통해서 트리가 학습되는 정도를 결정할 수 있고, 이는 모델의 최대 depth를 지정하는것으로 실행 가능하다.

```python
dtd = DecisionTreeClassifier(random_state=42, max_depth=3)
dtd.fit(input_train, target_train)

dtd.score(input_train, target_train)
# output: 0.8454877814123533

dtd.score(input_test, target_test)
# output: 0.8415384615384616
```

이전에 비해 고른 결과를 통해 과대적합이 해결된 것을 볼 수 있다.

이를 시각화 해 보면

![tree3.png](https://raw.githubusercontent.com/RFLXN/PnP.AI.2023/main/img/4-1/tree3.png)

위와 같고, 혼자 주황색으로 표시되는 노드를 따라가 보면
* sugar > 1.625
* alcohol <= 11.025
를 만족하는 샘플만 화이트 와인으로 예측한다는 것을 알 수 있다.
