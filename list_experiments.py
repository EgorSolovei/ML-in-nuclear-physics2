from processing_one_experiment import start_data_processing

experiments_list = {
    1: {'name_exp': 'exp_1',
        'path_from': r'D:\Programming\Science\csv_data',
        'path_to': r'D:\Programming\Science\new_version',
        'begin_number_file': 1,
        'end_number_file': 100,
        'borders': {0: [0, 1], 1: [1, 16.4]},
        'sensor_dist': [400, 500, 600],
        'subrings': [5, 10, 15, 20]},

    2: {'name_exp': 'exp_2',
        'path_from': r'D:\Programming\Science\csv_data',
        'path_to': r'D:\Programming\Science\new_version',
        'begin_number_file': 1,
        'end_number_file': 100,
        'borders': {0: [0, 1], 1: [1, 16.4]},
        'sensor_dist': [400, 500, 600],
        'subrings': [5, 10, 15, 20]},
}

for number_experiment, params_experiment in experiments_list.items():
    experiment_name = experiments_list[number_experiment]['name_exp']
    path_to = experiments_list[number_experiment]['path_to']
    path_from = experiments_list[number_experiment]['path_from']

    begin_number_file = experiments_list[number_experiment]['begin_number_file']
    end_number_file = experiments_list[number_experiment]['end_number_file']

    class_borders = experiments_list[number_experiment]['borders']
    sensor_dist = experiments_list[number_experiment]['sensor_dist']
    subrings = experiments_list[number_experiment]['subrings']

    print(f"Начата обработка данных эксперимента {experiment_name}")
    start_data_processing(path_from=path_from, path_to=path_to, start_file=1, end_file=100,
                          borders=class_borders, dist_sensor=sensor_dist, subrings=subrings)

