import struct
import numpy as np
import torch
import matplotlib.pyplot as plt


def parse_spx(file_path):
    """
    Парсит .spx файл и возвращает данные двух спектров и их длины волн
    """
    with open(file_path, 'rb') as f:
        file_content = f.read()

    # Данные спектров: 2 датчика по 4096 отсчетов (2 байта на отсчет)
    sensor_data_bytes = file_content[:16384]
    sensor_data = struct.unpack('<8192H', sensor_data_bytes)
    sensor_data_0 = np.array(sensor_data[:4096], dtype=np.float32)
    sensor_data_1 = np.array(sensor_data[4096:], dtype=np.float32)

    # Параметры MSCALE для калибровки длины волны
    params_start = 0xC006
    params_block = file_content[params_start:]

    pixels_0, waves_0 = [], []
    pixels_1, waves_1 = [], []

    for i in range(0, len(params_block), 24):
        param_bytes = params_block[i:i + 24]
        if len(param_bytes) < 24:
            break

        name = param_bytes[:15].decode('ascii', errors='ignore').strip('\x00').strip()
        param_type = param_bytes[15]
        if param_type != 0x44:  # Тип double
            continue

        value = struct.unpack('<d', param_bytes[16:24])[0]

        if name.startswith('MSCALE_0_'):
            pixels_0.append(int(name.split('_')[-1]))
            waves_0.append(value)
        elif name.startswith('MSCALE_1_'):
            pixels_1.append(int(name.split('_')[-1]) - 4096)
            waves_1.append(value)

    pixels_arr_0 = np.arange(4096)
    pixels_arr_1 = np.arange(4096)

    wavelengths_0 = np.interp(pixels_arr_0, pixels_0, waves_0)
    wavelengths_1 = np.interp(pixels_arr_1, pixels_1, waves_1)

    return sensor_data_0, sensor_data_1, wavelengths_0, wavelengths_1


def parse_spx_to_tensor_with_mask(file_path):
    """
    Объединяет спектры с NaN-разрывом и возвращает тензор и маску валидных значений
    """
    sensor_0, sensor_1, wave_0, wave_1 = parse_spx(file_path)

    wavelengths = np.concatenate((wave_0, [np.nan], wave_1))
    values = np.concatenate((sensor_0, [np.nan], sensor_1))

    combined = np.stack((wavelengths, values), axis=1)
    tensor = torch.tensor(combined, dtype=torch.float32)

    mask = ~torch.isnan(tensor[:, 1])  # True для валидных значений

    return tensor, mask


def parse_spx_to_split_tensor(file_path):
    """
    Преобразует данные двух спектров в один двухканальный тензор
    """
    sensor_data_0, sensor_data_1, wavelengths_0, wavelengths_1 = parse_spx(file_path)
    stacked = np.stack([sensor_data_0, sensor_data_1], axis=0)
    tensor = torch.tensor(stacked, dtype=torch.float32)

    return tensor, wavelengths_0, wavelengths_1


def plot_tensor_with_mask(tensor):
    """
    Визуализирует спектр с NaN-разрывом
    """
    wavelengths = tensor[:, 0].numpy()
    values = tensor[:, 1].numpy()
    plt.plot(wavelengths, values)
    plt.grid(False)
    # plt.show()


def plot_split_tensor(tensor, wavelengths_0, wavelengths_1):
    """
    Визуализирует спектр со split-каналом
    """
    plt.plot(wavelengths_0, tensor[0].numpy())
    plt.plot(wavelengths_1, tensor[1].numpy())
    plt.grid(False)
    # plt.show()
