import numpy
from sklearn.neighbors import KNeighborsClassifier
from data import fish_length, fish_weight

fish_data = [[ln, wt] for ln, wt in zip(fish_length, fish_weight)]
fish_target = ([1] * 35) + ([0] * 14)

input_data = numpy.array(fish_data)
target_data = numpy.array(fish_target)


def shuffle(seed: int, length: int) -> numpy.ndarray:
    numpy.random.seed(seed)
    arr = numpy.arange(length)
    numpy.random.shuffle(arr)

    return arr


index = shuffle(42, len(fish_data))

train_idx = index[:35]
input_train = input_data[train_idx]
target_train = target_data[train_idx]

test_idx = index[35:]
input_test = input_data[test_idx]
target_test = target_data[test_idx]

kn = KNeighborsClassifier()

kn.fit(input_train, target_train)

score = kn.score(input_test, target_test)

predict = kn.predict(input_test)

print(score)
print(predict)

print(target_test)
