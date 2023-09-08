import pandas as pd
import numpy as np

# чтобы не вылазило предупреждение, с которым я потом как-нибудь разберусь
pd.options.mode.chained_assignment = None


def get_vector_feature(idx_event, data, borders, R, r_cut, all_r, dist):
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

    # data_for_vec нужно преобразовать в вектор признаков. data - обрезанные входные данные
    data, data_for_vec = split_data_event(data, idx_event)

    # Фильтруем данные
    # -----------------------------------------------------------------

    # больше нам эти колонки не нужны, так как мы уже сделали классификацию по прицельному параметру
    data_for_vec.drop(columns=['lepton_number', 'strangeness', 'type_of_particles'], inplace=True)
    data_for_vec.query("particle_charge != 0", inplace=True)

    # Импульс
    # -----------------------------------------------------------------
    def total_impulse(row):
        total = np.power(row.impulse_x, 2) + np.power(row.impulse_y, 2) + np.power(row.impulse_z, 2)
        return np.sqrt(total)

    data_for_vec['modul_sum_impulse'] = data_for_vec.apply(total_impulse, axis=1)  # тут pandas ругается на что-то

    # Прилетит ли частица на какой-либо датчик
    # -----------------------------------------------------------------
    alpha_0 = np.arctan(r_cut / dist[-1])  # Минимальный угол попадания
    beta_0 = np.arctan(R / dist[0])  # Максимальный угол попадания

    def to_sensor(data):
        arccos = np.arccos(data.impulse_z / data.modul_sum_impulse)
        # положительное положение датчиков
        result_plus = arccos.between(alpha_0, beta_0, inclusive="neither")
        # отрицательное положение датчиков
        result_minus = arccos.between(np.pi - beta_0, np.pi - alpha_0, inclusive="neither")
        return result_plus | result_minus

    data_for_vec['to_any_sensor'] = to_sensor(data_for_vec)  # тут pandas ругается на что-то
    data_for_vec.query("to_any_sensor == True", inplace=True)  # убираем те частицы, которые не долетаю ни до одного датчика

    # Уберём колонки, которые уже не пригодятся
    # здесь каждая частица долетит до какого-нибудь датчика. Поэтому to_any_sensor уже не нужна
    data_for_vec.drop(columns=["particle_charge", "to_any_sensor"], inplace=True)

    # На какой датчик прилетела частица
    # -----------------------------------------------------------------
    alpha_i = [np.arctan(r_cut / dist_i) for dist_i in dist]  # минимальные углы для i датчика
    beta_i = [np.arctan(R / dist_i) for dist_i in dist]  # максимальный угол для i датчика

    # создадим колонку с арккосинусом угла тета
    data_for_vec["arccos_theta"] = np.arccos(data_for_vec.impulse_z / data_for_vec.modul_sum_impulse)

    # передаём арккосинус угла и направление датчика
    def between_corner(arccos_series, direct):
        if direct == 1:
            return arccos_series.between(alpha_i[j], beta_i[j])
        elif direct == -1:
            return arccos_series.between(np.pi - beta_i[j], np.pi - alpha_i[j])

    # создаём колонку для каждого датчика и делаем проверку по углам.
    for j in range(len(dist)):
        name_column = "sensor_" + str(j + 1)
        data_for_vec[name_column] = 0

        data_for_vec.loc[between_corner(data_for_vec.arccos_theta, 1), name_column] = 1
        data_for_vec.loc[between_corner(data_for_vec.arccos_theta, -1), name_column] = -1

    # Разбиение на сектора
    # -----------------------------------------------------------------

    """
    Определяем номер кольца попадания частицы для каждого датчика
    Внешний цикл создаёт колонку для каждой пары датчика, в которой лежат нули.
    Внутренний цикл проверяет в какое кольцо на датчике прилетает частица [1, 2, 3, 4] - 4 кольца
    Функция во внутреннем цикле работает с отбором по углам
    Работает так, как и задумывалось, потому что если частица попала на датчик, то есть номер кольца
    попадания, в случае, если частица не попала на датчик, то как и ожидалось, там 0
    """

    def between_corner_for_ring(arccos_series, direct):  # нужно подумать над углами в between
        if direct == 1:
            return arccos_series.between(alpha_ring[j], alpha_ring[j + 1])
        elif direct == -1:
            return arccos_series.between(np.pi - alpha_ring[j + 1], np.pi - alpha_ring[j])

    # Разбиение по подкольцам
    # -----------------------------------------------------------------
    for i in range(len(dist)):
        name_column = "number_of_ring_" + str(i)  # Нумерация колец тоже будет с 0
        data_for_vec[name_column] = 0

        for j in range(len(all_r) - 1):
            alpha_ring = [np.arctan(r / dist[i]) for r in all_r]

            # сделаем нумерацию подколец с 0, чтобы было удобнее определять сектор на датчике
            data_for_vec.loc[between_corner_for_ring(data_for_vec.arccos_theta, 1), name_column] = j
            data_for_vec.loc[between_corner_for_ring(data_for_vec.arccos_theta, -1), name_column] = j

    """
    Цикл подобен коду выше, только теперь это делается для части окружности
    Условия по которым идёт отбор:
    1) Частица должна прилететь на какой-то датчик: data_for_vec[column_sensor] != 0
    2) Отбор по знаку impulse_x, так как это определяет нужную половину датчика (правую или левую)
    3) И проверка на какой сектор прилетает частица: between_angle
    """

    def between_angle(left_border, rigth_border):
        result = data_for_vec.arctg_phi.between(left_border, rigth_border)
        return result

    def condition(column, sign_impulse_x, left_angle, rigth_angle):
        if sign_impulse_x == 1:
            return (data_for_vec[column] != 0) & (data_for_vec.impulse_x > 0) & (between_angle(left_angle, rigth_angle))
        elif sign_impulse_x == -1:
            return (data_for_vec[column] != 0) & (data_for_vec.impulse_x < 0) & (between_angle(left_angle, rigth_angle))

    data_for_vec["arctg_phi"] = np.arctan(data_for_vec.impulse_y / data_for_vec.impulse_x)

    for i in range(len(dist)):
        name_column = "number_of_piece_" + str(i + 1)  # str(i + 1) - пара датчиков. Значения там - 1 ... 8 - сектор прилёта
        column_sensor = "sensor_" + str(i + 1)  # для какого сенсора проверяем условия
        data_for_vec[name_column] = 0

        data_for_vec.loc[condition(column_sensor, 1, 0, np.pi / 4), name_column] = 1
        data_for_vec.loc[condition(column_sensor, 1, np.pi / 4, np.pi / 2), name_column] = 2
        data_for_vec.loc[condition(column_sensor, 1, -np.pi / 2, -np.pi / 4), name_column] = 7
        data_for_vec.loc[condition(column_sensor, 1, -np.pi / 4, 0), name_column] = 8

        data_for_vec.loc[condition(column_sensor, -1, -np.pi / 2, -np.pi / 4), name_column] = 3
        data_for_vec.loc[condition(column_sensor, -1, -np.pi / 4, 0), name_column] = 4
        data_for_vec.loc[condition(column_sensor, -1, 0, np.pi / 4), name_column] = 5
        data_for_vec.loc[condition(column_sensor, -1, np.pi / 4, np.pi / 2), name_column] = 6

    # алгоритм определения сектора по кольцу попадания и куску по фи
    data_for_vec["sector_1"] = data_for_vec.number_of_ring_0 * 8 + data_for_vec.number_of_piece_1
    data_for_vec["sector_2"] = data_for_vec.number_of_ring_1 * 8 + data_for_vec.number_of_piece_2
    data_for_vec["sector_3"] = data_for_vec.number_of_ring_2 * 8 + data_for_vec.number_of_piece_3

    # уберём все вспомогательные колонки
    data_for_vec.drop(columns=['number_of_ring_1', 'number_of_ring_2', 'number_of_ring_2',
                               'number_of_piece_1', 'number_of_piece_2', 'number_of_piece_3',
                               'arctg_phi', 'arccos_theta'], inplace=True)

    # Определение времени пролёта частицы
    # -----------------------------------------------------------------
    data_for_vec["velocity"] = data_for_vec.impulse_z / np.sqrt(data_for_vec.mass ** 2 + data_for_vec.modul_sum_impulse ** 2)

    """
    Нужно обрабатывать случаи, когда частица не прилетает на датчик
    умножим расстояние на индикатор прилёта частицы - 1, -1 или 0.
    в случае, если частица НЕ прилетела на датчик, то и время пролёта будет 0
    возьмём модуль, так как для времен не имеет значение, в каком направлении летит частица
    """
    data_for_vec["time_1"] = abs((dist[0] * data_for_vec.sensor_1) / data_for_vec.velocity)
    data_for_vec["time_2"] = abs((dist[1] * data_for_vec.sensor_2) / data_for_vec.velocity)
    data_for_vec["time_3"] = abs((dist[2] * data_for_vec.sensor_3) / data_for_vec.velocity)

    # уберём лишнее и сделаем в приятном виде
    data_for_vec.drop(columns=["velocity", "impulse_x", "impulse_y", "impulse_z", "modul_sum_impulse", "mass"], inplace=True)

    data_for_vec = data_for_vec[["sensor_1", "sector_1", "time_1",
                                 "sensor_2", "sector_2", "time_2",
                                 "sensor_3", "sector_3", "time_3", "class_param"]]

    # Создание результирующего вектора
    # -----------------------------------------------------------------
    lst_col = ['class']
    for i in range(1, len(dist) + 1):  # len(dist) - количество пар датчиков
        for j in range(1, 32 + 1):
            lst_col.append(f'sns_{i}_sct_{j}_plus')
            lst_col.append(f'sns_{i}_sct_{j}_minus')

    res = pd.Series([0] * 193, index=lst_col)

    # заполнение вектора res
    def position_in_res_vec(df):
        res['class'] = df.class_param.values[0]
        for i in range(1, len(dist) + 1):  # len(dist) - количество пар датчиков
            for j in range(1, 33):  # тут нужно смотреть на сколько секторов мы разбиваем.
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

    # Обработка случая, когда после фильтрации не осталось никаких частиц попавших на датчик. В этом случае вернём нулевой вектор
    if data_for_vec.shape[0] == 0:
        return res.values, data
    else:
        position_in_res_vec(data_for_vec)
        return res.values, data
