import json
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


# Определяем наследующий класс от torch.utils.data.Dataset для подготовки датасета
class PreprocessedDataset(Dataset):
    def __init__(self, csv_file):
        # На вход подается путь к csv файлу, выбираются необходимые колонки
        self.data = pd.read_csv(csv_file, usecols=[3, 4], header=None, names=['image_path', 'label'],
                                skiprows=1)  # Загружаем CSV-файл

        # print(self.data.head()) # Проверка выбранных колонок

        # Создание словаря для маппинга меток
        self.label_map = {label: idx for idx, label in enumerate(self.data['label'].unique())}

        # Сохранение словаря
        # with open('labels.json', 'w') as f:
        # json.dump(self.label_map, f)

        # Преобразование меток в числовые значения
        self.data['label'] = self.data['label'].map(self.label_map)

        # Определение трансформации
        self.transform = transforms.ToTensor()

    def __len__(self):
        # Возвращаем количество строк в CSV
        return len(self.data)

    def __getitem__(self, idx):
        # Извлекаем путь к изображению из первой колонки CSV
        image_path = self.data.iloc[idx, 0]

        # Извлекаем метки
        label = int(self.data.iloc[idx, 1])

        # Загружаем изображение
        image = Image.open(image_path).convert('RGB')  # Конвертация в RGB на всякий случай
        image = self.transform(image)  # Преобразуем в тензор

        # Возвращаем изображение и параметры как тензор
        return image, label
