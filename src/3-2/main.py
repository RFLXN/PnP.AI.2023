from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from numpy import unique
from matplotlib.pyplot import plot, xlabel, ylabel, show
from data import fish_data

input_data = fish_data[["Weight", "Length", "Diagonal", "Height", "Width"]].to_numpy()
target_data = fish_data["Species"].to_numpy()

input_train, input_test, target_train, target_test = train_test_split(input_data, target_data, random_state=42)

ss = StandardScaler()
ss.fit(input_train, target_train)

input_train_scaled = ss.transform(input_train)
input_test_scaled = ss.transform(input_test)

sc = SGDClassifier(loss="log_loss", max_iter=10, random_state=42)
sc.fit(input_train_scaled, target_train)

sc_train_score = sc.score(input_train_scaled, target_train)
sc_test_score = sc.score(input_test_scaled, target_test)

print(sc_train_score)
print(sc_test_score)

sc.partial_fit(input_train_scaled, target_train)

sc_partial_train_score = sc.score(input_train_scaled, target_train)
sc_partial_test_score = sc.score(input_test_scaled, target_test)

print(sc_partial_train_score)
print(sc_partial_test_score)

sc_100 = SGDClassifier(loss="log_loss", max_iter=100, random_state=42, tol=None)
sc_100.fit(input_train_scaled, target_train)

score_train_100 = sc_100.score(input_train_scaled, target_train)
score_test_100 = sc_100.score(input_test_scaled, target_test)

print(score_train_100)
print(score_test_100)
