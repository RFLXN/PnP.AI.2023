from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from matplotlib.pyplot import scatter, plot, xlabel, ylabel, show
from numpy import column_stack, arange
from data import perch_length, perch_weight


input_train, input_test, target_train, target_test = train_test_split(perch_length, perch_weight, random_state=42)

input_train = input_train.reshape(-1, 1)
input_test = input_test.reshape(-1, 1)

kn = KNeighborsRegressor(n_neighbors=3)
kn.fit(input_train, target_train)

predict = kn.predict([[50]])
print(predict)

"""
distances, indexes = kn.kneighbors([[1000]])

scatter(input_train, target_train)
scatter(input_train[indexes], target_train[indexes], marker="D")
scatter(1000, 1033, marker="^")

xlabel("length")
ylabel("weight")

show()
"""

lr = LinearRegression()
lr.fit(input_train, target_train)

lr_predict = lr.predict([[50]])
print(lr_predict)

print(lr.coef_, lr.intercept_)

"""
scatter(input_train, target_train)
plot([15, 50], [15 * lr.coef_ + lr.intercept_, 50 * lr.coef_ + lr.intercept_])
scatter(50, 1241.8, marker="^")
xlabel("length")
ylabel("weight")
show()
"""

input_train_poly = column_stack((input_train**2, input_train))
input_test_poly = column_stack((input_test**2, input_test))

poly_lr = LinearRegression()
poly_lr.fit(input_train_poly, target_train)

poly_predict = poly_lr.predict([[50**2, 50]])
print(poly_predict)

print(poly_lr.coef_, poly_lr.intercept_)

point = arange(15, 50)
scatter(input_train, target_train)
plot(point, 1.01 * point**2 - 21.6 * point + 116.05)
scatter(50, 1574, marker="^")
xlabel("length")
ylabel("weight")
show()
