import pandas as pd
import numpy as np
from copy import copy
from imblearn.over_sampling import SMOTE
import my_functions as mf  # это моя функция по обработке признаков


def main_code():
    sample_set = pd.read_excel(r'C:\STUDY\SkillFactory\Module_6\kaggle\auto_dataset_short.xlsx',
                               sheet_name='test')
    submission_set = copy(sample_set['sell_id'])  # оставляем id's для файла submission
    auto_data = pd.read_excel(r'C:\STUDY\SkillFactory\Module_6\kaggle\auto_dataset_short.xlsx',
                              sheet_name='auto')

    # записи без цены не несут для нас ценности
    auto_data.dropna(subset=["price"], inplace=True)

    # оставляю только бренды, которые есть в target data-set
    sample_brands = sample_set.brand.unique()
    # тут некрасиво, hard-coding в split('-') под конкретный brand Mercedes-Benz
    auto_data.brand = auto_data.brand.apply(lambda row: row.split('-')[0].upper())
    auto_data = auto_data[auto_data['brand'].isin(sample_brands)]

    # удалим 'Владение' - слищком много пропусков, под 70% всего объема
    auto_data.drop(['Владение'], axis=1, inplace=True)
    sample_set.drop(['Владение'], axis=1, inplace=True)

    # предварительно проверил количество строк с пропусками. Их были едницы на сете в 88 тыс строк
    auto_data.dropna(inplace=True)

    # выравниваем признаки в двух дата-сетах
    auto_data = mf.putting_order(auto_data)
    sample_set = mf.putting_order(sample_set)

    # удалим выбросы в training датасете
    value_list = ['price', 'mileage', 'car_age', 'enginePower']
    auto_data = mf.outlayers(auto_data, value_list)

    # удаляем лишние признаки
    auto_data = mf.remove_useless(auto_data)
    sample_set = mf.remove_useless(sample_set)

    # делаем дамми категриальных признаков
    auto_data = pd.get_dummies(auto_data,
                               columns=['bodyType', 'fuelType', 'color', 'vendor'])  # оставляем brand для oversampling
    sample_set = pd.get_dummies(sample_set, columns=['bodyType', 'brand', 'fuelType', 'color', 'vendor'])


    # сделаем oversampling по признаку brand, используем SMOTE
    sampler = SMOTE(random_state=0)
    sampl_cat = auto_data['brand']
    sampl_set = auto_data.drop(columns=['brand'], axis=1)
    auto_data_set, auto_data_cat = sampler.fit_resample(sampl_set, sampl_cat)  # SMOTE сеплинг
    auto_data = pd.concat([auto_data_set, auto_data_cat], axis=1)

    # закaнчиваем dummy в auto-set
    auto_data = pd.get_dummies(auto_data, columns=['brand'])
    #sample_set = pd.get_dummies(sample_set, columns=['brand'])

    # после dummy выравниваем количество колонок между сетами
    for column_item in sample_set.columns:
        if column_item not in auto_data.columns:
            auto_data[column_item] = 0

    return auto_data, sample_set, submission_set
