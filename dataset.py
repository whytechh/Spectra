import pandas as pd
import torch
from torch.utils.data import Dataset
import json


# Определяем наследующий класс от torch.utils.data.Dataset для подготовки датасета
class PreprocessedDataset(Dataset):
    def __init__(self, csv_file=None, df=None, label_map=None):

        # Если на вход подается csv-файл
        if csv_file is not None:
            self.data = pd.read_csv(csv_file, usecols=[3, 4], header=None, names=['image_path', 'label'], skiprows=1)
            # print(self.data.head()) # Проверка выбранных колонок

        # Если на вход подается датафрейм (для валидации)
        elif df is not None:
            self.data = df.copy()

        # Проверка на случай подачи csv_file и df одновременно
        elif csv_file is not None and df is not None:
            raise ValueError('Укажите либо csv_file, либо df')

        # Очистка меток
        self.data.dropna(subset=['label'], inplace=True)  # Удалить строки с NaN метками
        self.data['label'] = self.data['label'].astype(str).str.strip()  # Привести все метки к строкам без пробелов

        # Создание словаря для маппинга меток
        if label_map is not None:
            self.label_map = label_map
        else:
            unique_labels = sorted(self.data['label'].unique())
            self.label_map = {label: idx for idx, label in enumerate(unique_labels)}

            # Сохраняем label_map только при его генерации
            with open('labels.json', 'w', encoding='utf-8') as f:
                json.dump(self.label_map, f, indent=2, ensure_ascii=False)

        # Преобразование меток в числовые значения
        self.data['label'] = self.data['label'].map(self.label_map)

    def __len__(self):
        # Возвращаем количество строк в CSV
        return len(self.data)

    def __getitem__(self, idx):
        # Извлекаем путь к изображению из первой колонки CSV
        image_path = self.data.iloc[idx, 0]

        # Извлекаем метки
        label = int(self.data.iloc[idx, 1])

        # Загружаем изображение
        image = torch.load(image_path + '.pt', weights_only=True)

        # Возвращаем изображение и параметры как тензор
        return image, label
