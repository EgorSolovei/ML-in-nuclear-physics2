import os
import yaml
import numpy as np
import pandas as pd

from grid_search_models import searchCV_model
from processing_one_experiment import experiment_processing


# функция проверки корректности параметров эксперимента.
def correct_params(start_file, end_file, dist, all_r, step_ang):
    if 1 <= start_file <= end_file <= 100 and (dist == sorted(dist)) \
            and (all_r == sorted(all_r)) and 0 < step_ang <= np.pi:
        return True
    else:
        return False


def join_processed_data(save_dir):
    # склеим все данные в один файл
    data = pd.DataFrame()
    for i in range(1, 101):
        temp_df = pd.read_csv(save_dir + f"/processed_data/processed_data{i}.csv")
        data = pd.concat([data, temp_df], ignore_index=True)
    data.to_csv(save_dir + "/data.csv", index=False)


def start_experiment_processing():
    list_experiments = None
    with open("experiment_configs.yaml", "r") as config:
        list_experiments = yaml.safe_load(config)

    for params_experiment in list_experiments:
        experiment_name = params_experiment['name_exp']
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
        os.mkdir(path_to)
        os.mkdir(path_to + "/processed_data")

        with open(".gitignore", 'a') as git_ignore:
            git_ignore.write("\n" + experiment_name)

        # запишем в файл параметры эксперимента
        with open(path_to + '/experiment_param.txt', 'w') as f:
            f.write(f"Название эксперимента: {experiment_name}.\nГраницы классов: {class_borders}."
                    f"\nРасстояние до датчиков: {sensor_dist}.\nПодкольца разбиения: {subrings}."
                    f"\nШаг угла: {step_angle}.\nКонфигурация данных: {data_mode}")

        if correct_params(begin_number_file, end_number_file, sensor_dist, subrings, step_angle):
            print(f"Параметры эксперимента {experiment_name} корректны. Началась обработка данных")
            experiment_processing(class_borders, subrings, sensor_dist, step_angle, path_from, path_to,
                                  begin_number_file, end_number_file, data_mode)
            if len(os.listdir(path_to + "/processed_data")) == 100:
                print(f"Все файлы эксперимента {experiment_name} обработаны.")
                join_processed_data(save_dir=path_to)
                # searchCV_model(name_exp=experiment_name, size_data=10000)  # поиск лучших гиперпараметров для алгоритмов
            else:
                print(f"Данные эксперимента {experiment_name} обработаны НЕ до конца!!!")
        else:
            print(f"Параметры эксперимента {experiment_name} НЕ корректны!!!")


if __name__ == "__main__":
    start_experiment_processing()