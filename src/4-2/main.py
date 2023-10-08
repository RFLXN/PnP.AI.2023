from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV, train_test_split, RandomizedSearchCV
from numpy import mean, max, arange
from scipy.stats import uniform, randint
from data import wine_data

input_data = wine_data[["alcohol", "sugar", "pH"]].to_numpy()
target_data = wine_data["class"].to_numpy()

dt = DecisionTreeClassifier(random_state=42)

score = cross_validate(dt, input_data, target_data)
print(score)
print(mean(score["test_score"]))

splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

score_fold = cross_validate(dt, input_data, target_data, cv=splitter)
print(score_fold)
print(mean(score_fold["test_score"]))

input_train, input_test, target_train, target_test = train_test_split(input_data, target_data, random_state=42)
"""
params = {"min_impurity_decrease": [0.0001, 0.0002, 0.0003, 0.0004]}

gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)

gs.fit(input_train, target_train)

print(gs.best_estimator_)
"""

"""
params = {
    "min_impurity_decrease": arange(0.0001, 0.001, 0.0001),
    "max_depth": arange(5, 20, 1),
    "min_samples_split": arange(2, 100, 10),
}

gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)

gs.fit(input_train, target_train)

print(gs.best_estimator_)
print(max(gs.cv_results_["mean_test_score"]))
"""

params = {
    "min_impurity_decrease": uniform(0.0001, 0.001),
    "max_depth": randint(20, 50),
    "min_samples_split": randint(2, 25),
    "min_samples_leaf": randint(1, 25),
}

rs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1, n_iter=100)

rs.fit(input_train, target_train)

print(rs.best_estimator_)
print(max(rs.cv_results_["mean_test_score"]))
