# ML in nuclear physics
  ## Цель
  Цель заключается в определении класса прицельного параметра столкновения двух ядер с помощью методов ML. 
  Изначально задача регрессии, но сведена к задаче классификации - 
  разбили отрезок возможных значений прицельного параметра на классы. 
  Есть некоторые ограничения на скорость работы алгоритма - 
  нужно предсказывать класс параметра за $50$ нс = $50 * 10^{-9}$ с
  
  ## Данные
  Данные созданы генератором - 100 файлов по 2000 столкновений (событий) в каждом. 
  Являются представлением того, куда и с какой скоростью разлетелись частицы после столкновения. 
  
  ## Файлы
  - data_preprocessing.ipynb - предобработка одного файла данных с подробным описанием.
  - list_experiment.py - задаются параметры нескольких экспериментов. Вызывается processing_one_experiment.py
  - processing_one_experiment.py - обработка сырых файлов и запускается create_data_loop
  - create_data_loop.py - преобразование одного события (столкновения) в вектор
  - data.csv - очищенные данные для обучения модели.