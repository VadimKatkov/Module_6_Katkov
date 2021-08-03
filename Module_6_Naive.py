import pandas as pd
import numpy as np
from copy import copy
import mainCode as mc
import my_functions as mf  # это моя функция по основным шагам features engineering

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge, ElasticNet, SGDRegressor


pd.set_option('display.max_columns', None)

# это запуcкает два файла функций c обработкой признаков и подготовкой дата-сетов
auto_data, sample_set, submission_set = mc.main_code()

Y = auto_data['price'].astype(int)
X = auto_data.drop(columns=['price'], axis=1)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42,
                                                    test_size=0.3, shuffle=True)

# тут эксперементируем со скалером
myScaler = RobustScaler()
# myScaler = StandardScaler()
X_train = myScaler.fit_transform(X_train)
X_test = myScaler.transform(X_test)
sample_set = myScaler.transform(sample_set)


# блок моделей
# 1. OLS
ols = LinearRegression()
ols.fit(X_train, Y_train)
predict_ols = ols.predict(X_test)
print('OLS:')
mf.my_model_metrics(Y_test, predict_ols)

# 2. Ridge
ridge = Ridge(alpha=0.5)
ridge.fit(X_train, Y_train)
predict_ridge = ridge.predict(X_test)
print('Ridge:')
mf.my_model_metrics(Y_test, predict_ridge)

# 3. Lasso
lasso = Lasso(alpha=0.5)
lasso.fit(X_train, Y_train)
predict_lasso = lasso.predict(X_test)
print('Lasso:')
mf.my_model_metrics(Y_test, predict_lasso)

# 4. Bayesian
bayesian = BayesianRidge()
bayesian.fit(X_train, Y_train)
predict_bayesian = bayesian.predict(X_test)
print('Bayesian:')
mf.my_model_metrics(Y_test, predict_bayesian)

# 5. ElasticNet
en = ElasticNet(alpha=0.01)
en.fit(X_train, Y_train)
predict_en = en.predict(X_test)
print('ElasticNet:')
mf.my_model_metrics(Y_test, predict_en)


# 6. SGDRegressor
en = SGDRegressor(alpha=0.01)
en.fit(X_train, Y_train)
predict_sgd = en.predict(X_test)
print('SGDRegressor:')
mf.my_model_metrics(Y_test, predict_sgd)

# отдельно определил уровень инфляции потребительских цен между sample_set & auto_data - считаю, что он 8%
submission_predict = ols.predict(sample_set)
submission_set = pd.concat([submission_set, pd.DataFrame(submission_predict)], names=['sell_id', 'price'], axis=1)
submission_set.set_index('sell_id', inplace=True)

submission_set.to_csv(r"C:\STUDY\SkillFactory\Module_6\kaggle\submission_naive.csv")
