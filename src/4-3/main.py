from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from numpy import mean
from data import wine_data

input_data = wine_data[["alcohol", "sugar", "pH"]].to_numpy()
target_data = wine_data["class"].to_numpy()

input_train, input_test, target_train, target_test = train_test_split(input_data, target_data, random_state=42)

rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)

scores = cross_validate(rf, input_train, target_train, return_train_score=True, n_jobs=-1)

print(mean(scores["train_score"]))
print(mean(scores["test_score"]))

rf.fit(input_train, target_train)
print(rf.oob_score_)

et = ExtraTreesClassifier(n_jobs=-1, random_state=42)

scores = cross_validate(et, input_train, target_train, return_train_score=True, n_jobs=-1)

print(mean(scores["train_score"]))
print(mean(scores["test_score"]))

gb = GradientBoostingClassifier(random_state=42)

scores = cross_validate(gb, input_train, target_train, return_train_score=True, n_jobs=-1)

print(mean(scores["train_score"]))
print(mean(scores["test_score"]))

hg = HistGradientBoostingClassifier(random_state=42)
scores = cross_validate(hg, input_train, target_train, return_train_score=True, n_jobs=-1)

print(mean(scores["train_score"]))
print(mean(scores["test_score"]))
