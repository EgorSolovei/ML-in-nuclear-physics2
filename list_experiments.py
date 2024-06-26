import numpy as np
import pandas as pd
import os

from processing_one_experiment import experiment_processing
from grid_search_models import searchCV_model


list_experiments = [
    {'name_exp': '4meters_8ring_4angle_with_light',
     'path_from': '/home/egor/programming/python/ML-in-nuclear-physics2/row_data',
     'path_to': '/home/egor/programming/python/ML-in-nuclear-physics2',
     'begin_number_file': 1,
     'end_number_file': 100,
     'borders': {1: [0, 1], 0: [1, 17]},
     'sensor_dist': [4],
     'subrings': [2.5, 5.31, 8.12, 10.93, 13.74, 16.55, 19.36, 22.17, 24.98],
     'step_angle': np.pi / 2,
     'data_mode': 'one_time'
     }
]


# функция проверки корректности параметров эксперимента.
def correct_params(start_file, end_file, dist, all_r, step_ang):
    if 1 <= start_file <= end_file <= 100 and (dist == sorted(dist)) \
            and (all_r == sorted(all_r)) and 0 < step_ang <= np.pi:
        return True
    else:
        return False


# склеим все данные в один файл
def join_processed_data():
    data = pd.DataFrame()
    for i in range(1, 101):
        temp_df = pd.read_csv(path_to + f"/processed_data/processed_data{i}.csv")
        data = pd.concat([data, temp_df], ignore_index=True)
    data.to_csv(path_to + "/data.csv", index=False)


for params_experiment in list_experiments:
    experiment_name = params_experiment['name_exp']

    # Номер начального и конечного файла обработки. Всего 100 файлов
    begin_number_file = params_experiment['begin_number_file']
    end_number_file = params_experiment['end_number_file']

    # Параметры эксперимента
    class_borders = params_experiment['borders']
    sensor_dist = params_experiment['sensor_dist']
    subrings = params_experiment['subrings']
    step_angle = params_experiment['step_angle']
    data_mode = params_experiment['data_mode']

    # Путь получения и сохранения данных
    path_from = params_experiment['path_from']
    path_to = params_experiment['path_to'] + "/" + experiment_name
    os.mkdir(path_to)  # создание папки эксперимента
    os.mkdir(path_to + "/processed_data")  # создание папки обработанных данных эксперимента

    # запишем в .gitignore папку с обработанными данными
    with open(".gitignore", 'a') as git_ignore:
        git_ignore.write("\n" + experiment_name)

    # запишем в файл параметры эксперимента
    with open(path_to + '/experiment_param.txt', 'w') as f:
        f.write(f"Название эксперимента: {experiment_name}.\nГраницы классов: {class_borders}."
                f"\nРасстояние до датчиков: {sensor_dist}.\nПодкольца разбиения: {subrings}."
                f"\nШаг угла: {step_angle}.\nКонфигурация данных: {data_mode}")

    # проверим корректность параметров
    if correct_params(begin_number_file, end_number_file, sensor_dist, subrings, step_angle):
        print(f"Параметры эксперимента {experiment_name} корректны. Началась обработка данных")
        experiment_processing(class_borders, subrings, sensor_dist, step_angle, path_from, path_to,
                              begin_number_file, end_number_file, data_mode)
        if len(os.listdir(path_to + "/processed_data")) == 100:
            print(f"Все файлы эксперимента {experiment_name} обработаны.")
            join_processed_data()  # склеим все данные в один файл data.csv
            # searchCV_model(name_exp=experiment_name, size_data=10000)  # поиск лучших гиперпараметров для алгоритмов
        else:
            print(f"Данные эксперимента {experiment_name} обработаны НЕ до конца!!!")
    else:
        print(f"Параметры эксперимента {experiment_name} НЕ корректны!!!")
