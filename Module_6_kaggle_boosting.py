import pandas as pd
import numpy as np
from copy import copy
import mainCode as mc
import my_functions as mf  # это моя функция по основным шагам features engineering
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

# это запуcкает два файла функций c обработкой признаков и подготовкой дата-сетов
auto_data, sample_set, submission_set = mc.main_code()

Y = auto_data['price'].astype(int)
X = auto_data.drop(columns=['price'], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42,
                                                    test_size=0.3, shuffle=True)

# тут эксперементируем со скалером
myScaler = RobustScaler()
#myScaler = StandardScaler()
X_train = myScaler.fit_transform(X_train)
X_test = myScaler.transform(X_test)
sample_set = myScaler.transform(sample_set)

myCatBoost = CatBoostRegressor(iterations=5000,
                               random_seed=42,
                               eval_metric='MAPE',
                               custom_metric=['R2', 'MAE'],
                               silent=True)

myCatBoost.fit(X_train, Y_train,
               eval_set=(X_test, Y_test),
               verbose_eval=0,
               use_best_model=True)

predict_boost = myCatBoost.predict(X_test)

print('CatBoost:')
mf.my_model_metrics(Y_test, predict_boost)


# отдельно определил уровень инфляции потребительских цен между sample_set & auto_data - считаю, что он 8%
submission_predict = myCatBoost.predict(sample_set) / 1.08
submission_set = pd.concat([submission_set, pd.DataFrame(submission_predict)], names=['sell_id', 'price'], axis=1)
submission_set.set_index('sell_id', inplace=True)

submission_set.to_csv(r"C:\STUDY\SkillFactory\Module_6\kaggle\submission_boost.csv")


