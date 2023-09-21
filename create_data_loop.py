import pandas as pd
import numpy as np

from create_name import create_column_names

# чтобы не вылазило предупреждение, с которым я потом как-нибудь разберусь
pd.options.mode.chained_assignment = None


def get_vector_feature(idx_event, data, borders, all_r, dist, step_angle):
    # Функция определения класса прицельного параметра
    def define_class_impact_param(impact_param):
        for i in range(len(borders)):
            if borders[i][0] <= impact_param < borders[i][1]:
                return i

    # Функция, которая берёт только одно событие - данные с одного столкновения ядер.
    def split_data_event(df, idx):
        for i in range(df.shape[0]):
            if df.mass[i] == -1 and df.particle_charge[i] == idx:
                cnt_particle = df.lepton_number[i]  # количество частиц в столкновении
                data_temp = df[i + 1: i + 1 + cnt_particle]  # вырезаем нужную часть

                class_param = define_class_impact_param(df.strangeness[i])  # класс прицельного параметра
                data_temp.insert(0, "class_param", class_param)

                # обрежем события, которые уже обработаны, чтобы не перебирать все строки с начала
                # хотя возможно и такое копирование будет делаться даже дольше и съедать память
                df = df[i + 1 + cnt_particle::].reset_index(drop=True)  # сбросили индексы
                return df, data_temp

    # to_vec нужно преобразовать в вектор признаков. data - обрезанные входные данные
    data, to_vec = split_data_event(data, idx_event)

    # Фильтруем данные
    # -----------------------------------------------------------------

    # больше нам эти колонки не нужны, так как мы уже сделали классификацию по прицельному параметру
    to_vec.drop(columns=['lepton_number', 'strangeness', 'type_of_particles'], inplace=True)
    to_vec.query("particle_charge != 0", inplace=True)

    # Импульс
    # -----------------------------------------------------------------
    def total_impulse(row):
        total = np.power(row.impulse_x, 2) + np.power(row.impulse_y, 2) + np.power(row.impulse_z, 2)
        return np.sqrt(total)

    to_vec['modul_sum_impulse'] = to_vec.apply(total_impulse, axis=1)  # тут pandas ругается на что-то

    # Прилетит ли частица на какой-либо датчик
    # -----------------------------------------------------------------
    # P.S dist[0] - радиус выреза датчика. dist[-1] - радиус датчика
    alpha_0 = np.arctan(all_r[0] / dist[-1])  # Минимальный угол попадания
    beta_0 = np.arctan(all_r[-1] / dist[0])  # Максимальный угол попадания

    def to_sensor(temp_data):
        arccos = np.arccos(temp_data.impulse_z / temp_data.modul_sum_impulse)
        # положительное положение датчиков
        result_plus = arccos.between(alpha_0, beta_0, inclusive="neither")
        # отрицательное положение датчиков
        result_minus = arccos.between(np.pi - beta_0, np.pi - alpha_0, inclusive="neither")
        return result_plus | result_minus

    to_vec['to_any_sensor'] = to_sensor(to_vec)  # тут pandas ругается на что-то
    # убираем те частицы, которые не долетаю ни до одного датчика
    to_vec.query("to_any_sensor == True", inplace=True)

    # Здесь каждая частица долетит до какого-нибудь датчика. Поэтому to_any_sensor уже не нужна
    to_vec.drop(columns=["particle_charge", "to_any_sensor"], inplace=True)

    # На какой датчик прилетела частица
    # -----------------------------------------------------------------
    alpha_i = [np.arctan(all_r[0] / dist_i) for dist_i in dist]  # минимальные углы для i датчика
    beta_i = [np.arctan(all_r[-1] / dist_i) for dist_i in dist]  # максимальные углы для i датчика

    # создадим колонку с арккосинусом угла тета
    to_vec["arccos_theta"] = np.arccos(to_vec.impulse_z / to_vec.modul_sum_impulse)

    # создаём колонку для каждого датчика и делаем проверку прилетит ли частица через этот датчик.
    for j in range(len(dist)):
        name_column = "sensor_" + str(j + 1)
        to_vec[name_column] = 0

        def between_corner(arccos_series, direct):  # передаём арккосинус угла и направление датчика
            if direct == 1:
                return arccos_series.between(alpha_i[j], beta_i[j])
            elif direct == -1:
                return arccos_series.between(np.pi - beta_i[j], np.pi - alpha_i[j])

        to_vec.loc[between_corner(to_vec.arccos_theta, 1), name_column] = 1
        to_vec.loc[between_corner(to_vec.arccos_theta, -1), name_column] = -1

    # Разбиение по подкольцам. До этого момента всё работает хорошо (вроде бы)
    # -----------------------------------------------------------------
    """
    Определяем номер кольца попадания частицы для каждого датчика
    Внешний цикл создаёт колонку для каждой пары датчика, в которой лежат нули.
    Внутренний цикл проверяет в какое кольцо на датчике прилетает частица [0, 1, ...., len(all_r) -1]
    Функция во внутреннем цикле работает с отбором по углам
    Работает так, как и задумывалось, потому что если частица попала на датчик, то есть номер кольца
    попадания, в случае, если частица не попала на датчик, то как и ожидалось, там 0
    """

    # Здесь начинаются странности с данными. Ни одна частица не прилетает в кольцо
    for i in range(len(dist)):
        name_column = "number_of_ring_" + str(i)
        to_vec[name_column] = 0
        alpha_ring = [np.arctan(r / dist[i]) for r in all_r]

        for j in range(len(all_r) - 1):  # т.к если границ n, то подколец n-1
            def between_corner_for_ring(arccos_series, direct):  # нужно подумать над углами в between
                if direct == 1:
                    return arccos_series.between(alpha_ring[j], alpha_ring[j + 1])
                elif direct == -1:
                    return arccos_series.between(np.pi - alpha_ring[j + 1], np.pi - alpha_ring[j])

            # сделаем нумерацию подколец с 0, чтобы было удобнее определять сектор на датчике
            to_vec.loc[between_corner_for_ring(to_vec.arccos_theta, 1), name_column] = j
            to_vec.loc[between_corner_for_ring(to_vec.arccos_theta, -1), name_column] = j

    # -----------------------------------------------------------------

    """
    Цикл подобен коду выше, только теперь это делается для части окружности
    Условия по которым идёт отбор:
    1) Частица должна прилететь на какой-то датчик: to_vec[sensor] != 0
    2) Отбор по знаку impulse_x, так как это определяет нужную половину датчика (правую или левую)
    3) И проверка на какой сектор прилетает частица: between_angle
    """

    def between_angle(left_border, right_border):
        result = to_vec.arctg_phi.between(left_border, right_border)
        return result

    def condition(column, sign_impulse_x, left_angle, right_angle):
        if sign_impulse_x == 1:
            return (to_vec[column] != 0) & (to_vec.impulse_x > 0) & (between_angle(left_angle, right_angle))
        elif sign_impulse_x == -1:
            return (to_vec[column] != 0) & (to_vec.impulse_x < 0) & (between_angle(left_angle, right_angle))

    to_vec["arctg_phi"] = np.arctan(to_vec.impulse_y / to_vec.impulse_x)
    pieces = int(2 * np.pi / step_angle)  # количество секторов на одном датчике

    for i in range(len(dist)):
        sensor = "sensor_" + str(i + 1)  # для какого сенсора проверяем условия
        name_column = "sector_id" + str(i)  # str(i) - пара датчиков. Значения там - сектор прилёта
        to_vec[name_column] = 0

        sector_id = 1
        start_ang = -np.pi / 2  # т.к область значений арктангенса [-pi/2, pi/2]
        for k in range(int(pieces / 2)):  # цикл для impulse_x > 0
            left_border_angle = start_ang + k * step_angle  # левая граница, угла сектора
            right_border_angle = start_ang + (k + 1) * step_angle  # правая граница, угла сектора
            to_vec.loc[condition(sensor, 1, left_border_angle, right_border_angle), name_column] = sector_id
            sector_id += 1

        for k in range(int(pieces / 2)):  # цикл для impulse_x < 0
            left_border_angle = start_ang + k * step_angle
            right_border_angle = start_ang + (k + 1) * step_angle
            to_vec.loc[condition(sensor, -1, left_border_angle, right_border_angle), name_column] = sector_id
            sector_id += 1

    # Алгоритм определения сектора по кольцу попадания и номеру части сектора.
    # Умножим на модуль сенсора - индикатор (возможные значения: -1, 0, 1) пролёта частицы через этот сенсор.
    for i in range(len(dist)):
        to_vec[f"sector_{i + 1}"] = abs(to_vec[f"sensor_{i + 1}"]) * to_vec[f"number_of_ring_{i}"] * pieces \
                                                                                    + to_vec[f"sector_id{i}"]
        to_vec.drop(columns=[f'number_of_ring_{i}', f'sector_id{i}'], inplace=True)

    # Определение времени пролёта частицы
    # -----------------------------------------------------------------
    to_vec["velocity"] = to_vec.impulse_z / np.sqrt(to_vec.mass ** 2 + to_vec.modul_sum_impulse ** 2)

    """
    Нужно обрабатывать случаи, когда частица не прилетает на датчик
    умножим расстояние на индикатор прилёта частицы - 1, -1 или 0.
    в случае, если частица НЕ прилетела на датчик, то и время пролёта будет 0
    возьмём модуль, так как для времен не имеет значение, в каком направлении летит частица
    """
    # цикл определения времени пролёта для каждой пары датчиков
    for i in range(1, len(dist) + 1):
        to_vec[f"time_{i}"] = abs((dist[i - 1] * to_vec[f"sensor_{i}"]) / to_vec.velocity)

    # Уберём все вспомогательные колонки.
    to_vec.drop(columns=["velocity", "impulse_x", "impulse_y",
                         "impulse_z", "modul_sum_impulse", "mass",
                         "arctg_phi", "arccos_theta"], inplace=True)

    # Создание результирующего вектора
    # -----------------------------------------------------------------
    lst_col = create_column_names(pair_sensor=len(dist), pieces=pieces, quan_rings=(len(all_r) - 1))
    res = pd.Series([0.0] * len(lst_col), index=lst_col)

    # заполнение вектора res
    def position_in_res_vec(df):
        res['class'] = df.class_param.values[0]
        for i in range(1, len(dist) + 1):  # len(dist) - количество пар датчиков
            for j in range(1, pieces + 1):  # тут нужно смотреть на сколько секторов мы разбиваем.
                # создадим условие для нахождения минимального времени пролёта
                condition_plus = (df[f'sensor_{i}'] == 1) & (df[f'sector_{i}'] == j)
                condition_minus = (df[f'sensor_{i}'] == -1) & (df[f'sector_{i}'] == j)

                # сортируем по времени и берём самое меньшее время
                min_data_plus = df[condition_plus].sort_values(f'time_{i}')
                min_data_minus = df[condition_minus].sort_values(f'time_{i}')

                # возможны случаи, когда ни одна частица не пролетела через датчик
                if min_data_plus.shape[0] != 0:
                    res[f'sns_{i}_sct_{j}_plus'] = min_data_plus[f'time_{i}'].values[0]
                if min_data_minus.shape[0] != 0:
                    res[f'sns_{i}_sct_{j}_minus'] = min_data_minus[f'time_{i}'].values[0]

    # Обработка случая, когда после фильтрации не осталось никаких частиц попавших на датчик.
    # В этом случае вернём нулевой вектор
    if to_vec.shape[0] == 0:
        return res.values, data
    else:
        position_in_res_vec(to_vec)
        return res.values, data
