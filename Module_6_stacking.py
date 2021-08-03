import pandas as pd
import numpy as np
from copy import copy
import mainCode as mc
import my_functions as mf
from sklearn.base import clone

from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from catboost import CatBoostRegressor

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

# cv = StratifiedKFold(n_splits=3)
cv = KFold(n_splits=10, shuffle=True, random_state=42)

estimators = [('cb', CatBoostRegressor(iterations=5000,
                                       random_seed=42,
                                       eval_metric='MAPE',
                                       silent=True)),
              ('rf', RandomForestRegressor(n_estimators=1000,
                                           n_jobs=-1,
                                           max_depth=15,
                                           max_features='log2',
                                           random_state=42,
                                           oob_score=True))]

myStackReg = StackingRegressor(estimators=estimators,
                               final_estimator=GradientBoostingRegressor(
                                   min_samples_split=2,
                                   learning_rate=0.03,
                                   max_depth=10,
                                   n_estimators=1000))

myStackReg.fit(X_train, Y_train)
predict_stack = myStackReg.predict(X_test)
print('Stacking:')
mf.my_model_metrics(Y_test, predict_stack)


# отдельно определил уровень инфляции потребительских цен между sample_set & auto_data - считаю, что он 8%
submission_predict = myStackReg.predict(sample_set) / 1.08
submission_set = pd.concat([submission_set, pd.DataFrame(submission_predict)], names=['sell_id', 'price'], axis=1)
submission_set.set_index('sell_id')

submission_set.to_csv(r"C:\STUDY\SkillFactory\Module_6\kaggle\submission_stacking.csv")

