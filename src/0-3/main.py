from sklearn.neighbors import KNeighborsClassifier
from data import bream_length, smelt_length, bream_weight, smelt_weight

length = bream_length + smelt_length
weight = bream_weight + smelt_weight

fish_data = [[ln, wt] for ln, wt in zip(length, weight)]

fish_target = ([1] * 35) + ([0] * 14)

kn = KNeighborsClassifier()
nb_kn = KNeighborsClassifier(n_neighbors=49)

kn.fit(fish_data, fish_target)
nb_kn.fit(fish_data, fish_target)

score_kn = kn.score(fish_data, fish_target)
score_nb = nb_kn.score(fish_data, fish_target)

print(score_kn)
print(score_nb)

predict_kn = kn.predict([[30, 600]])
predict_nb = nb_kn.predict([[30, 600]])

print(predict_kn)
print(predict_nb)
