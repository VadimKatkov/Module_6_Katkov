import pandas as pd
import numpy as np
from copy import copy
from sklearn.metrics import mean_absolute_percentage_error, \
    mean_absolute_error, \
    mean_squared_error, r2_score
from tqdm import tqdm


# функция приводит в порядок дата-сеты
def putting_order(df):
    # оставляем только название кузова без признака кол-ва дверей и пр
    df['bodyType'] = df['bodyType'].apply(lambda row: row.split()[0])

    # переводим объем двигателя в int
    df['engineDisplacement'] = df['engineDisplacement'].apply \
        (lambda row: float(0.0) if row.split()[0] == 'LTR' else float(
            row.split()[0]))

    # перводим мощность двигателя в int
    df['enginePower'] = df['enginePower'].apply(lambda row: int(row.split(' ')[0]))

    # формируем период авто в эксплуатации
    df['unix_year'] = pd.to_datetime(df['parsing_unixtime'].astype(int), unit='s').dt.year
    df['car_age'] = df.apply(lambda row: row['unix_year'] - row['modelDate'], axis=1)

    # переводим в int количество владельцев автомобиляю Думаю тут есть приоритетность
    df['Владельцы'] = df['Владельцы'].apply(lambda row: int(row[0]))  # кол-во владельцев в int

    # кодируем привод. Наивысший приоритет у полного привода, потом передний и потом задний
    drive_weels = {'полный': 3, 'передний': 2, 'задний': 1}
    df['Привод'] = df['Привод'].map(drive_weels)

    # кодируем трансмиссию. Считаю, есть приоритетность для покупателя
    transmission = {'вариатор': 4, 'автоматическая': 3, 'роботизированная': 2, 'механическая': 1}
    df['vehicleTransmission'] = df['vehicleTransmission'].map(transmission)

    # подрежем milage к более укрупненным цифрам, до 10 000
    df['mileage'] = df['mileage'].apply(lambda row: int(round(row, -4)))

    # просто приводим тип данных к единому формату
    # df[['modelDate', 'productionDate']] = df[['modelDate', 'productionDate']].astype(int)

    # добавление этого признака ухудшило метрику
    # df['equipment_list'] = list(map(lambda item: item.count(':') if isinstance(item, str) else 0, df['equipment_dict']))

    return df


# функция удаяет ненужны колонки
def remove_useless(df):
    '''
    После просмотров дата-сетов удаляю следубщие колонки:
    1. car_url - 100% заполнение. Содержание самого url не даст какой-либо полезной инфо
    2. complectation_dict - заполнено на 18%. Слишком много пропусков.
    3. image - 100% заполнение. Обрабатывать фото пока не умеем
    4. Владение - 35% заполнения в в этом наборе и 25% в auto_data. Слишком много пропусков.
    5. sell_id - уникальный идентификатор. Просто не нужен
    6. Состояние - принимает одно значение в обоих дата-сетах: "не требует ремонта"
    7. priceCurrency - принимает одно значение в обоих дата-сетах: "RUB"
    8. Таможня - принимает одно значение в обоих дата-сетах: "Растаможен"
    9. model_info - не добавляет полезной информации
    10.description - ну это просто поток сознания. Не известно что с ним делать
    11.equipment_dict - удалил после подсчета оборудвания
    12.model_info - удалил, чать информации дублируется, в части нет ничего полезного
    13.name  - удалить, дублирует model name
    14.super_gen - удалить
    15.parsing_unixtime - удаляем после парсинга
    16.vehicleConfiguration - дублирует данные
    17.model_name - категория полезная но слишком много уникальных значений и нет гарнтий
    18.numberOfDoors - при анализе значимости категории было в самом низу
    19. ПТС - фигня
    20.Руль - при анализе значимости категории было в самом низу
    21. productionDate - удалил так высокая корреляция м modelDate
    22. modelDate - удалил после добавления car_age
    '''
    columns_to_delete = ["car_url", "image", "complectation_dict", 'unix_year', 'modelDate',
                         "Состояние", "priceCurrency", "Таможня", "model_info",
                         "equipment_dict", "super_gen", "parsing_unixtime", 'name',
                         'model_name', "ПТС", "Руль", 'vehicleConfiguration',
                         "description", "sell_id", "numberOfDoors", 'productionDate']
    df.drop(columns=columns_to_delete, inplace=True)
    return df


# функция обработки выбросов. применяем классический подход - 25%, 75% и +/-1.5 интерквартиля
def outlayers(df, value_list):
    for value in value_list:
        for brand_item in df['brand'].unique():
            quantile_1 = np.quantile(df[df.brand == brand_item][value], 0.25, interpolation='midpoint')
            quantile_3 = np.quantile(df[df.brand == brand_item][value], 0.75, interpolation='midpoint')
            iqt = quantile_3 - quantile_1
            df.drop(df[(df['brand'] == brand_item) & (df[value] < quantile_1 - 1.5 * iqt) |
                       (df[value] > quantile_3 + 1.5 * iqt)].index)

    return df

# функция расчета и вывода метрик
def my_model_metrics(Y_test, y_predict):
    MSE = round(mean_squared_error(Y_test, y_predict), 0)
    print('-MSEroot-:', round(np.sqrt(MSE), 0))
    print(f'---MAPE--: {mean_absolute_percentage_error(Y_test, y_predict) * 100:0.2f}%')
    print('---R2 ---:', round(r2_score(Y_test, y_predict), 3))
    print()
