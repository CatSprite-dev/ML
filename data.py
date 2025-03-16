from sklearn import datasets
import pandas as pd

data = datasets.load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)

from sklearn.model_selection import train_test_split

x = df[['sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']]
y = df['age']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)
first_estimate = pd.DataFrame([y_test.values, lr.predict(x_test)])

from sklearn.metrics import mean_squared_error, r2_score

print(f"Среднеквадратичная ошибка тестовой выборки: {mean_squared_error(y_test, lr.predict(x_test))}")
print(f"Среднеквадратичная ошибка обучающей выборки: {mean_squared_error(y_train, lr.predict(x_train))}")
print(f"Коэффициент детерминации тестовой выборки: {r2_score(y_test, lr.predict(x_test))}")
print(f"Коэффициент детерминации обучающей выборки: {r2_score(y_train, lr.predict(x_train))}")
#print(first_estimate.T)
#print(data.DESCR)
