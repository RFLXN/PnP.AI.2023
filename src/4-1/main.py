import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib.pyplot import figure, show
from data import wine_data

data = wine_data[["alcohol", "sugar", "pH"]].to_numpy()
target = wine_data["class"].to_numpy()

input_train, input_test, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(input_train, target_train)

train_score = dt.score(input_train, target_train)
test_score = dt.score(input_test, target_test)

print(train_score)
print(test_score)

"""
figure(figsize=(10, 7))
plot_tree(dt)
plt.show()
"""
"""
figure(figsize=(10, 7))
plot_tree(dt, max_depth=1, filled=True, feature_names=["alcohol", "sugar", "pH"])
show()
"""

dtd = DecisionTreeClassifier(random_state=42, max_depth=3)
dtd.fit(input_train, target_train)

train_score_depth = dtd.score(input_train, target_train)
test_score_depth = dtd.score(input_test, target_test)

print(train_score_depth)
print(test_score_depth)

figure(figsize=(20, 15))
plot_tree(dtd, filled=True, feature_names=["alcohol", "sugar", "pH"])
show()
