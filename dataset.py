# import json
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


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
        
        # Создание словаря для маппинга меток
        if label_map is not None:
            self.label_map = label_map
        else:
            self.label_map = {label: idx for idx, label in enumerate(self.data['label'].unique())}
        
        # Сохранение словаря (сохранять только при первом запуске)
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
