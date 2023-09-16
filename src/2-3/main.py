from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from matplotlib.pyplot import plot, xlabel, ylabel, show
from numpy import log10
from data import perch_weight

perch_data = read_csv("./data.csv").to_numpy()
print(perch_data)

input_train, input_test, target_train, target_test = train_test_split(perch_data, perch_weight, random_state=42)

pf = PolynomialFeatures(include_bias=False)
pf.fit(input_train)
input_train_poly = pf.transform(input_train)
input_test_poly = pf.transform(input_test)

lr = LinearRegression()
lr.fit(input_train_poly, target_train)

train_score = lr.score(input_train_poly, target_train)
print(train_score)

test_score = lr.score(input_test_poly, target_test)
print(test_score)

pf_over = PolynomialFeatures(degree=5, include_bias=False)
pf_over.fit(input_train)
over_input_train_poly = pf_over.transform(input_train)
over_input_test_poly = pf_over.transform(input_test)

print(over_input_train_poly.shape)
lr_over = LinearRegression()
lr_over.fit(over_input_train_poly, target_train)

over_train_score = lr_over.score(over_input_train_poly, target_train)
print(over_train_score)

over_test_score = lr_over.score(over_input_test_poly, target_test)
print(over_test_score)

ss = StandardScaler()
ss.fit(over_input_train_poly)

input_train_scaled = ss.transform(over_input_train_poly)
input_test_scaled = ss.transform(over_input_test_poly)

ridge = Ridge()
ridge.fit(input_train_scaled, target_train)

ridge_train_score = ridge.score(input_train_scaled, target_train)
print(ridge_train_score)

ridge_test_score = ridge.score(input_test_scaled, target_test)
print(ridge_test_score)

train_scores = []
test_scores = []

alpha_values = [0.001, 0.01, 1, 10, 100]

for alpha in alpha_values:
    current_ridge = Ridge(alpha=alpha)
    current_ridge.fit(input_train_scaled, target_train)
    train_scores.append(current_ridge.score(input_train_scaled, target_train))
    test_scores.append(current_ridge.score(input_test_scaled, target_test))

"""
plot(log10(alpha_values), train_scores)
plot(log10(alpha_values), test_scores)
xlabel("alpha")
ylabel("R^2")
show()
"""

alpha_ridge = Ridge(alpha=0.1)
alpha_ridge.fit(input_train_scaled, target_train)

alpha_ridge_train_score = alpha_ridge.score(input_train_scaled, target_train)
print(alpha_ridge_train_score)

alpha_ridge_test_score = alpha_ridge.score(input_test_scaled, target_test)
print(alpha_ridge_test_score)

lasso = Lasso()
lasso.fit(input_train_scaled, target_train)

lasso_train_score = lasso.score(input_train_scaled, target_train)
print(lasso_train_score)

lasso_test_score = lasso.score(input_test_scaled, target_test)
print(lasso_test_score)

lasso_train_scores = []
lasso_test_scores = []

for alpha in alpha_values:
    current_lasso = Lasso(alpha=alpha)
    current_lasso.fit(input_train_scaled, target_train)
    lasso_train_scores.append(current_lasso.score(input_train_scaled, target_train))
    lasso_test_scores.append(current_lasso.score(input_test_scaled, target_test))

plot(log10(alpha_values), lasso_train_scores)
plot(log10(alpha_values), lasso_test_scores)
xlabel("alpha")
ylabel("R^2")
show()

alpha_lasso = Lasso(alpha=10)
alpha_lasso.fit(input_train_scaled, target_train)

alpha_lasso_train_score = alpha_lasso.score(input_train_scaled, target_train)
print(alpha_lasso_train_score)

alpha_lasso_test_score = alpha_lasso.score(input_test_scaled, target_test)
print(alpha_lasso_test_score)
