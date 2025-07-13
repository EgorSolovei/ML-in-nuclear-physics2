def create_column_names(pair_sensor, pieces, quan_rings):
    """
    :param pair_sensor: количество пар сенсоров
    :param pieces: количество секторов разбиения
    :param quan_rings: количество подколец
    :return: возвращается список имён, длиной 2 * pair_sensor * piece * quan_rings + 1
    """
    col_name = ['impact_prm', 'class']
    for i in range(1, pair_sensor + 1): 
        for j in range(1, pieces * quan_rings + 1):  # от 1 до количества секторов на одном датчике
            col_name.append(f'sns_{i}_sct_{j}_plus')
            col_name.append(f'sns_{i}_sct_{j}_minus')
    return col_name
