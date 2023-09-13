from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from data import perch_length, perch_weight

input_train, input_test, target_train, target_test = train_test_split(perch_length, perch_weight, random_state=42)

input_train = input_train.reshape(-1, 1)
input_test = input_test.reshape(-1, 1)

kn = KNeighborsRegressor()
kn.fit(input_train, target_train)

score_test = kn.score(input_test, target_test)
print(score_test)

score_input = kn.score(input_train, target_train)
print(score_input)

few_kn = KNeighborsRegressor(n_neighbors=3)
few_kn.fit(input_train, target_train)

few_score_test = few_kn.score(input_test, target_test)
print(few_score_test)

few_score_input = few_kn.score(input_train, target_train)
print(few_score_input)
