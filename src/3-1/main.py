from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from numpy import round
from data import fish_data

input_data = fish_data[["Weight", "Length", "Diagonal", "Height", "Width"]].to_numpy()
target_data = fish_data["Species"].to_numpy()

input_train, input_test, target_train, target_test = train_test_split(input_data, target_data, random_state=42)

ss = StandardScaler()
ss.fit(input_train, target_train)

input_train_scaled = ss.transform(input_train)
input_test_scaled = ss.transform(input_test)

kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(input_train_scaled, target_train)

kn_train_score = kn.score(input_train_scaled, target_train)
kn_test_score = kn.score(input_test_scaled, target_test)

print(kn_train_score)
print(kn_test_score)

kn_predict = kn.predict(input_test_scaled[:5])
print(kn_predict)

kn_proba = kn.predict_proba(input_test_scaled[:5])
print(kn_proba)

print(kn.classes_)

kn_distances, kn_indexes = kn.kneighbors(input_test_scaled[3:4])
print(target_train[kn_indexes])

lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(input_train_scaled, target_train)

lr_train_score = lr.score(input_train_scaled, target_train)
lr_test_score = lr.score(input_test_scaled, target_test)

print(lr_train_score)
print(lr_test_score)

print(lr.predict(input_test_scaled[:5]))

lr_proba = lr.predict_proba(input_test_scaled[:5])
print(round(lr_proba, decimals=3))

print(lr.classes_)

print(lr.coef_.shape, lr.intercept_.shape)
