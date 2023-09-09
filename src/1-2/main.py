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

scatter(input_train[:, 0], input_train[:, 1])
scatter(25, 150, marker="^")
xlabel("length")
ylabel("weight")
show()
