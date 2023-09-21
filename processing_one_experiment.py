import time
import pandas as pd
import numpy as np

from create_data_loop import get_vector_feature
from create_name import create_column_names


def data_processing(borders, subrings, sensor_dist, step_angle, path_from, path_to, start_file=1, end_file=100):
    # на каждой итерации обрабатывается 1 файл и создаётся 1 файл готовых данных
    for file_i in range(start_file, end_file + 1):
        print(f"Начата обработка файла №{file_i}")
        start_time = time.time()  # засечём время

        data = pd.read_csv(path_from + f'/Data_{file_i}.csv', sep=';', index_col=False)

        # убрали не нужные колонки
        data.drop(columns=['baryon_number', 'impulse_z_lab', 'param_5'], inplace=True)

        # заменим все NaN, предполагая, что такого значения в массе нет, так как она должна быть положительна
        data.fillna(-1, inplace=True)

        # количество столкновений в файле (на случай, если их будет не постоянное количество)
        quan_event = data.query('mass == -1').agg({'mass': 'count'}).values[0]

        # создание пустого DataFrame с нужными колонками
        sectors = int(2 * np.pi / step_angle)  # количество секторов на одном датчике
        lst_col = create_column_names(len(sensor_dist), sectors, (len(subrings) - 1))
        data_res = pd.DataFrame(columns=lst_col)  # результирующий DataFrame, который записывается в файл

        # цикл создания файла (2000 строк)
        for id in range(1, quan_event + 1):
            vec, data = get_vector_feature(id, data, borders, subrings, sensor_dist, step_angle)
            data_res.loc[len(data_res.index)] = vec  # добавление вектора в конец

        print(f"Время обработки: {(time.time() - start_time):.02f} секунд")

        data_res.to_csv(path_to + f"/processed_data/processed_data{file_i}.csv", index=False)
        print("Результат записан\n")
