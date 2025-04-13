import gc
import struct
import numpy as np
import matplotlib.pyplot as plt


def parse_and_save_png_by_spx(file_path, save_path):
    """
    Функция для парсинга бинарного файла .spx и визуализации спектров

    Аргументы:
    file_path (str): Путь к файлу .spx

    Возвращает:
    Отображает график спектра, сохраняет картинку
    """
    # Открываем файл в бинарном режиме и читаем содержимое
    with open(file_path, 'rb') as f:
        file_content = f.read()

    # Считываем данные спектров (2 датчика по 4096 отсчетов, каждый отсчет занимает 2 байта)
    sensor_data_bytes = file_content[:16384]  # Первые 16384 байта — данные датчиков
    sensor_data = struct.unpack('<8192H', sensor_data_bytes)  # Распаковка в массив из 8192 значений

    # Разделяем данные на два датчика
    sensor_data_0 = sensor_data[:4096]  # Первый датчик
    sensor_data_1 = sensor_data[4096:]  # Второй датчик

    # Читаем параметры MSCALE с адреса 0xC006
    params_start = 0xC006
    params_block = file_content[params_start:]  # Берем данные после значения 0xC006

    # Переменные для хранения пикселей и соответствующих им длин волн для интерполяции
    pixels_for_interp_0, wavelengths_for_interp_0 = [], []
    pixels_for_interp_1, wavelengths_for_interp_1 = [], []

    # Разбираем параметры файла, каждый параметр занимает 24 байта
    for i in range(0, len(params_block), 24):
        param_bytes = params_block[i:i + 24]  # Извлекаем 24 байта параметра

        # Имя параметра
        name = param_bytes[:15].decode('ascii', errors='ignore').strip('\x00').strip()

        # Тип параметра (16-й байт)
        param_type = param_bytes[15]

        # Значение параметра (последние 8 байт)
        value_bytes = param_bytes[16:24]

        # Проверяем, является ли параметр типом double
        if param_type == 0x44:
            value = struct.unpack('<d', value_bytes)[0]
        else:
            continue

        # Проверяем, относится ли параметр к привязке длин волн (MSCALE) для первого или второго датчика
        if name.startswith('MSCALE_0_'):
            pixel_index = int(name.split('_')[-1])  # Извлекаем номер пикселя
            pixels_for_interp_0.append(pixel_index)
            wavelengths_for_interp_0.append(value)

        elif name.startswith('MSCALE_1_'):
            pixel_index = int(name.split('_')[-1])  # Номер пикселя
            pixels_for_interp_1.append(pixel_index - 4096)  # Корректируем индекс для второго датчика
            wavelengths_for_interp_1.append(value)

    # Создаем массив пикселей для интерполяции
    pixels_0 = np.arange(4096)
    pixels_1 = np.arange(4096)

    # Интерполируем длины волн для каждого датчика
    wavelengths_0 = np.interp(pixels_0, pixels_for_interp_0, wavelengths_for_interp_0)
    wavelengths_1 = np.interp(pixels_1, pixels_for_interp_1, wavelengths_for_interp_1)

    # Визуализация спектров
    fig = plt.figure(num=1, figsize=(2.24, 2.24), clear=True)
    ax = fig.add_subplot()

    ax.plot(wavelengths_0, sensor_data_0, linewidth=0.7, label='Датчик 0')
    ax.plot(wavelengths_1, sensor_data_1, linewidth=0.7, label='Датчик 1')

    ax.grid(False)  # Включаем сетку
    fig.tight_layout()  # Улучшаем расположение элементов
    plt.axis('off')

    # Сохраняем изображение спектра
    fig.savefig(save_path)
    fig.clear()
    plt.close(fig)
    gc.collect()
