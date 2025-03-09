import pandas as pd

def add_datetime_features(dataframe):
    """
    Добавляет новые временные признаки в таблицу с данными о поездках.

    Аргументы:
    dataframe (pd.DataFrame) - входной датафрейм, содержащий столбец 'pickup_datetime'.

    Возвращает:
    pd.DataFrame - датафрейм с добавленными столбцами:
        - 'pickup_date' (дата начала поездки, без времени)
        - 'pickup_hour' (час начала поездки)
        - 'pickup_day_of_week' (номер дня недели, где 0 - понедельник, 6 - воскресенье)
    """

    # Преобразуем столбец pickup_datetime в формат datetime, если он еще не в нем
    dataframe['pickup_datetime'] = pd.to_datetime(dataframe['pickup_datetime'])
    # Извлекаем дату (без времени)
    dataframe['pickup_date'] = dataframe['pickup_datetime'].dt.date
    # Извлекаем час дня, в который была начата поездка
    dataframe['pickup_hour'] = dataframe['pickup_datetime'].dt.hour
    # Извлекаем порядковый номер дня недели (с понедельник по воскресенье)
    dataframe['pickup_day_of_week'] = dataframe['pickup_datetime'].dt.day_name()
    
    return dataframe

def add_holiday_features(taxi_data, holiday_data):
    """
    Добавляет признак 'pickup_holiday' к таблице с данными о поездках. 
    Признак указывает, является ли день поездки праздничным (1 - праздничный, 0 - нет).
    
    Аргументы:
    taxi_data (pandas.DataFrame): таблица с информацией о поездках такси. Должна содержать столбец 'pickup_date' в формате datetime.
    holiday_data (pandas.DataFrame): таблица с информацией об известных праздничных днях. Должна содержать столбец 'date' в формате datetime.
    
    Возвращает: 
    pandas.Series: новый столбец 'pickup_holiday' в таблице taxi_data, указывающий, является ли день поездки праздничным (1 - праздничный, 0 - нет).
    """
    
    # Преобразование столбца 'pickup_date' такси дата в формат datetime, если он еще не в нем без времени
    taxi_data['pickup_date'] = pd.to_datetime(taxi_data['pickup_date'])
    
    # Преобразование столбца 'date' праздничных дней в формат datetime, если он еще не в нем
    holiday_data['date'] = pd.to_datetime(holiday_data['date'])
    
    # Создаем новый столбец 'pickup_holiday' в таблице с данными о поездках, 
    # где мы проверяем содержится ли дата поездки (taxi_data['pickup_date']) во фрейме 'date' праздничного дня (holiday_data). 
    # Если таковая есть, то выставляем 1 (да), иначе - 0 (нет)
    taxi_data['pickup_holiday'] = taxi_data['pickup_date'].isin(holiday_data['date']).astype(int)
    
    # Возвращаем новый столбец
    return taxi_data
    
def add_osrm_features(taxi_data, osrm_data):
    """
    Добавляет признаки из OSRM в таблицу поездок такси.

    Аргументы:
    taxi_data (pd.DataFrame): данные о поездках (должны содержать id, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude).
    osrm_data (pd.DataFrame): данные OSRM (должны содержать id, total_distance, total_travel_time, number_of_steps).

    Возвращает:
    pd.DataFrame: обновленный taxi_data с добавленными колонками total_distance, total_travel_time, number_of_steps.
    """
    
    merged_data = taxi_data.merge(
        osrm_data[['id', 'total_distance', 'total_travel_time', 'number_of_steps']],
        on='id',
        how='left'
    )
    
    return merged_data

import numpy as np

def get_haversine_distance(lat1, lng1, lat2, lng2):
    """
    Вычисляет расстояние между двумя точками на Земле по формуле Хаверсина.

    Аргументы:
    lat1, lng1 (float): широта и долгота первой точки (в градусах).
    lat2, lng2 (float): широта и долгота второй точки (в градусах).

    Возвращает:
    float: расстояние между точками в километрах.
    """

    # Переводим координаты из градусов в радианы
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    
    # Радиус Земли в километрах
    EARTH_RADIUS = 6371 

    # Разница широт и долгот
    lat_delta = lat2 - lat1
    lng_delta = lng2 - lng1

    # Вычисляем расстояние по формуле Хаверсина
    d = np.sin(lat_delta * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng_delta * 0.5) ** 2
    h = 2 * EARTH_RADIUS * np.arcsin(np.sqrt(d))  # Итоговое расстояние

    return h

def get_angle_direction(lat1, lng1, lat2, lng2):
    """
    Вычисляет направление (азимут) движения из первой точки во вторую.

    Аргументы:
    lat1, lng1 (float): широта и долгота начальной точки (в градусах).
    lat2, lng2 (float): широта и долгота конечной точки (в градусах).

    Возвращает:
    float: угол направления движения (азимут) в градусах, от -180 до 180.
    """

    # Переводим координаты из градусов в радианы
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    # Разница долгот в радианах
    lng_delta_rad = lng2 - lng1

    # Вычисляем компоненты угла по формуле угла пеленга
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)

    # Вычисляем угол направления и переводим его в градусы
    alpha = np.degrees(np.arctan2(y, x))

    return alpha

def add_geographical_features(taxi_data):
    """
    Добавляет географические признаки в таблицу поездок такси.

    Функция рассчитывает два новых признака:
    - 'haversine_distance' – кратчайшее расстояние между точками посадки и высадки (в километрах), используя формулу Хаверсина.
    - 'direction' – угол направления движения (азимут) в градусах, от -180 до 180.

    Аргументы:
    taxi_data (pd.DataFrame): Датафрейм с данными о поездках такси.
        Должен содержать следующие столбцы:
        - 'pickup_latitude' (float) – широта начальной точки.
        - 'pickup_longitude' (float) – долгота начальной точки.
        - 'dropoff_latitude' (float) – широта конечной точки.
        - 'dropoff_longitude' (float) – долгота конечной точки.

    Возвращает:
    pd.DataFrame: Обновленный датафрейм с добавленными столбцами:
        - 'haversine_distance' (float) – расстояние между точками (км).
        - 'direction' (float) – угол направления движения (градусы).
    """
    
    # Вычисляем расстояние Хаверсина
    taxi_data['haversine_distance'] = taxi_data.apply(
        lambda row: get_haversine_distance(
        row['pickup_latitude'], row['pickup_longitude'],
        row['dropoff_latitude'], row['dropoff_longitude']
        ), axis=1
    )

    # Вычисляем угол направления движения
    taxi_data['direction'] = taxi_data.apply(
        lambda row: get_angle_direction(
        row['pickup_latitude'], row['pickup_longitude'],
        row['dropoff_latitude'], row['dropoff_longitude']
        ), axis=1
    )
    
    return taxi_data


def add_cluster_features(taxi_data, kmeans):
    """
    Добавляет географические кластеры к таблице поездок на основе обученного KMeans.

    Аргументы:
    taxi_data (pd.DataFrame): Датафрейм с данными о поездках, содержащий столбцы:
        - 'pickup_latitude' (float): широта начальной точки.
        - 'pickup_longitude' (float): долгота начальной точки.
        - 'dropoff_latitude' (float): широта конечной точки.
        - 'dropoff_longitude' (float): долгота конечной точки.
    kmeans (KMeans): Обученная модель KMeans для географической кластеризации.

    Возвращает:
    pd.DataFrame: Обновленный taxi_data с добавленным столбцом:
        - 'geo_cluster' (int): номер географического кластера, к которому относится поездка.
    """

    taxi_data['geo_cluster'] = kmeans.predict(taxi_data[['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']])
    return taxi_data

def add_weather_features(taxi_data, weather_data):
    weather_data['date'] = pd.to_datetime(weather_data['date']).dt.date

    merge_data = taxi_data.merge(
        weather_data,
        left_on=['pickup_date', 'pickup_hour'],
        right_on=['date', 'hour'],
        how='left'
        )
    
    return merge_data

def fill_null_weather_data(taxi_data):
    """
    Заполняет пропущенные значения в данных о поездках.

    1. Пропуски в погодных данных ('temperature', 'visibility', 'wind_speed', 'precip'):
       - Заполняются медианными значениями в каждой группе 'pickup_date'.
       - Если в группе нет значений, применяется глобальная медиана столбца.
    2. Пропуски в 'events' заполняются строкой 'None' (отсутствие погодных явлений).
    3. Пропуски в данных OSRM ('total_distance', 'total_travel_time', 'number_of_steps'):
       - Заполняются медианными значениями по каждому столбцу.

    Аргументы:
    taxi_data (pd.DataFrame): Датафрейм с данными о поездках.

    Возвращает:
    pd.DataFrame: Обновленный датафрейм taxi_data с заполненными пропусками.
    """

    # Столбцы с числовыми погодными условиями
    weather_cols = ['temperature', 'visibility', 'wind_speed', 'precip']

    # Заполняем медианной по 'pickup_date', если вся группа NaN — заполняем глобальной медианой
    for col in weather_cols:
        taxi_data[col] = taxi_data[col].fillna(
            taxi_data.groupby('pickup_date')[col].transform('median')
        )
        taxi_data[col] = taxi_data[col].fillna(taxi_data[col].median())  # Заполняем глобальной медианой

    # Заполнение пропусков в 'events' строкой 'None'
    taxi_data['events'] = taxi_data['events'].fillna('None')

    # Столбцы с информацией из OSRM API
    osrm_cols = ['total_distance', 'total_travel_time', 'number_of_steps']

    # Заполняем медианной по каждому столбцу
    for col in osrm_cols:
        taxi_data[col] = taxi_data[col].fillna(taxi_data[col].median())

    return taxi_data

ALPHA = 0.05 # Наш установленный уровень значимости

# функция для принятия решения об отклонении нулевой гипотезы
def decision_normality(p):
    """Данная функция отображает сравнивает полученное значения (p) и в зависимости 
    от установленного уровня значимости (ALPHA) выдает варианты ответа:
    
    - Распределение не является нормальным
    - Распределение является нормальным

    Args:
        p (int()): уровень значения
    """
    print(f'p-value = {round(p, 2)}\n')
    
    if p <= ALPHA: 
        print(f'p-значение меньше, чем заданный уровень значимости {ALPHA}. Распределение отлично от нормального')
    else:
        print(f'p-значение больше, чем заданный уровень значимости {ALPHA}. Распределение является нормальным')


# функция для принятия решения об отклонении нулевой гипотезы
def decision_hypothesis(p):
    """Данная функция отображает сравнивает полученное значения (p) и в зависимости 
    от установленного уровня значимости (ALPHA) выдает варианты ответа:
    
    - Отвергаем нулевую гипотезу в пользу альтернативной
    - Нет основания отвергнуть нулевую гипотезу

    Args:
        p (int()): уровень значения
    """
    print(f'p-value = {round(p, 2)}\n')
    
    if p <= ALPHA:
        print(f'p-значение меньше, чем заданный уровень значимости {ALPHA}. Отвергаем нулевую гипотезу в пользу альтернативной.')
    else:
        print(f'p-значение больше, чем заданный уровень значимости {ALPHA}. У нас нет оснований отвергнуть нулевую гипотезу.')