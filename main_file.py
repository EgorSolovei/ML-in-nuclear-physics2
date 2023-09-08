import time
import pandas as pd
from create_data_loop import get_vector_feature

# параметры эксперимента
# -----------------------------

# пути для импорта и экспорта данных
path_from = r'D:\Programming\Science\csv_data'
path_to = r'D:\Programming\Science\new_version'

# номера файлов, которые нужно обработать
begin_number_file = 1
end_number_file = 2


borders = {0: [0, 1], 1: [1, 16.4]}  # границы классов
R = 25  # внешний радиус датчика (см)
r_cut = 5  # внутренний радиус датчика (см)
dist = [400, 500, 600]  # расстояние до соответствующего датчика (см)
all_r = [r_cut, 10, 15, 20, R]  # Разбиение на кольца датчика. Лучше ближе к центру чаще, а дальше - реже


# функция проверки корректности параметров эксперимента.
def correct_params():
    if begin_number_file < end_number_file and (dist == sorted(dist)) and (all_r == sorted(all_r)):
        return True
    else:
        return False


# функция создающая имена колонок для DataFrame
def create_column_names(pair_sensor):
    col_name = ['class']
    for i in range(1, pair_sensor + 1):  # pair_sensor - количество пар датчиков
        for j in range(1, 32 + 1):
            col_name.append(f'sns_{i}_sct_{j}_plus')
            col_name.append(f'sns_{i}_sct_{j}_minus')
    return col_name


def data_processing():
    # на каждой итерации обрабатывается 1 файл и создаётся 1 файл готовых данных
    for file_i in range(begin_number_file, end_number_file + 1):
        print(f"Начата обработка файла №{file_i}")
        start_time = time.time()  # засечём время

        data = pd.read_csv(path_from + f'\Data_{file_i}.csv', sep=';', index_col=False)

        # убрали пока не нужные колонки
        data.drop(columns=['baryon_number', 'impulse_z_lab', 'param_5'], inplace=True)

        # заменим все NaN, предполагая, что такого значения в массе нет, так как она должна быть положительна
        data.fillna(-1, inplace=True)

        # количество столкновений в файле (на случай, если их будет не постоянное количество)
        quan_event = data.query('mass == -1').agg({'mass': 'count'}).values[0]

        # создание пустого DataFrame с нужными колонками
        data_res = pd.DataFrame(columns=lst_col)  # результирующий DataFrame, который записывается в файл

        # цикл создания файла (2000 строк)
        for id in range(1, quan_event + 1):
            vec, data = get_vector_feature(idx_event=id, data=data, borders=borders,
                                           R=R, r_cut=r_cut, all_r=all_r, dist=dist)
            data_res.loc[len(data_res.index)] = vec  # добавление вектора в конец
            if id % 500 == 0:
                print(f"Выполнено до события №{id}")

        print(f"Время обработки файла №{file_i}: {(time.time() - start_time):.02f} секунд")

        data_res.to_csv(path_to + fr"\result_data{file_i}.csv", index=False)
        print("Результат записан\n")


if correct_params():
    print("Параметры эксперимента корректны!")
    lst_col = create_column_names(pair_sensor=len(dist))
    data_processing()
else:
    print("Параметры эксперимента некорректны!")
