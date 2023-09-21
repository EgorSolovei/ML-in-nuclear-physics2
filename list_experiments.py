import numpy as np
import pandas as pd
import os
from processing_one_experiment import data_processing


list_experiments = [
    {'name_exp': 'test_exp_1',
     'path_from': '/home/egor/programming/python/ML-in-nuclear-physics2/csv_data',
     'path_to': '/home/egor/programming/python/ML-in-nuclear-physics2',
     'begin_number_file': 1,
     'end_number_file': 100,
     'borders': {1: [0, 1], 0: [1, 16.4]},
     'sensor_dist': [4],
     'subrings': [5, 25],
     'step_angle': np.pi / 2
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

    # Путь получения и сохранения данных
    path_from = params_experiment['path_from']
    path_to = params_experiment['path_to'] + "/" + experiment_name
    os.mkdir(path_to)  # создание папки эксперимента
    os.mkdir(path_to + "/processed_data")  # создание папка обработанных данных эксперимента

    # запишем в .gitignore папку с обработанными данными
    with open(".gitignore", 'a') as git_ignore:
        git_ignore.write("\n" + experiment_name)

    with open(path_to + '/experiment_param.txt', 'w') as f:
        f.write(f"Название эксперимента: {experiment_name}.\nГраницы классов: {class_borders}."
                f"\nРасстояние до датчиков: {sensor_dist}.\nПодкольца разбиения: {subrings}.\nШаг угла: {step_angle}")

    # проверим корректность параметров
    if correct_params(begin_number_file, end_number_file, sensor_dist, subrings, step_angle):
        print(f"Параметры эксперимента {experiment_name} корректны. Началась обработка данных")
        data_processing(class_borders, subrings, sensor_dist, step_angle,
                        path_from, path_to, begin_number_file, end_number_file)
        # здесь написать функцию, которая будет склеивать все обработанные данные в папке
        if len(os.listdir(path_to + "/processed_data")) == 100:
            print(f"Все файлы эксперимента {experiment_name} обработаны.")
            join_processed_data()
        else:
            print(f"Данные эксперимента {experiment_name} обработаны НЕ до конца!!!")
    else:
        print(f"Параметры эксперимента {experiment_name} НЕ корректны!!!")
