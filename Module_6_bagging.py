import pandas as pd
import numpy as np
from copy import copy
import mainCode as mc
import my_functions as mf  # это моя функция по основным шагам features engineering

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor

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


bagging_model = BaggingRegressor(random_state=42, n_estimators=1000).fit(X_train, Y_train)
predict_bagging = bagging_model.predict(X_test)
print('Bagging:')
mf.my_model_metrics(X_test, Y_test)

# отдельно определил уровень инфляции потребительских цен между sample_set & auto_data - считаю, что он 8%
submission_predict = bagging_model.predict(sample_set) /1.08
submission_set = pd.concat([submission_set, pd.DataFrame(submission_predict)], names=['sell_id', 'price'], axis=1)
submission_set.set_index('sell_id', inplace=True)

submission_set.to_csv(r"C:\STUDY\SkillFactory\Module_6\kaggle\submission_bagging.csv")
