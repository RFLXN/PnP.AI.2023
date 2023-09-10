import numpy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.pyplot import scatter, xlabel, ylabel, show
from data import fish_length, fish_weight

fish_input = numpy.column_stack((fish_length, fish_weight))
fish_target = numpy.concatenate((numpy.ones(35), numpy.zeros(14)))

input_train, input_test, target_train, target_test = train_test_split(
    fish_input, fish_target, random_state=42, stratify=fish_target
)

kn = KNeighborsClassifier()
kn.fit(input_train, target_train)

distances, indexes = kn.kneighbors([[25, 150]])

print(input_train[indexes])
print(distances)

"""
scatter(input_train[:, 0], input_train[:, 1])
scatter(25, 150, marker="^")
scatter(input_train[indexes, 0], input_train[indexes, 1], marker="D")
xlabel("length")
ylabel("weight")
show()
"""

mean = numpy.mean(input_train, axis=0)
std = numpy.std(input_train, axis=0)
scaled_input_train = (input_train - mean) / std

scaled_input_test = (input_test - mean) / std

new = ([25, 150] - mean) / std

print(scaled_input_train[:, 0])

st_kn = KNeighborsClassifier()
st_kn.fit(scaled_input_train, target_train)

st_score = st_kn.score(scaled_input_test, target_test)
print(st_score)

st_predict = st_kn.predict([new])
print(st_predict)

st_distances, st_indexs = st_kn.kneighbors([new])

scatter(scaled_input_train[:, 0], scaled_input_train[:, 1])
scatter(new[0], new[1], marker="^")
scatter(scaled_input_train[st_indexs, 0], scaled_input_train[st_indexs, 1], marker="D")
xlabel("length")
ylabel("weight")
show()
