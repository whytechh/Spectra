import json
import numpy as np
import torch
from spectra_preprocessing import parse_spx_to_split_tensor, parse_spx_to_tensor_with_mask
from torch.utils.data import Dataset


class SpectraDatasetSplitChannels(Dataset):
    def __init__(self, file_paths, labels=None, transform=None):
        self.file_paths = file_paths
        self.transform = transform

        if labels is None:
            raise ValueError("Необходимо передать список меток для каждого файла.")

        if len(file_paths) != len(labels):
            raise ValueError("Количество путей и меток должно совпадать.")

        self.raw_labels = [str(label).strip() for label in labels]

        # Создаем label_map
        unique_labels = sorted(set(self.raw_labels))
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}

        # Сохраняем label_map
        with open('labels.json', 'w', encoding='utf-8') as f:
            json.dump(self.label_map, f, indent=2, ensure_ascii=False)

        # Числовые метки
        self.numeric_labels = [self.label_map[label] for label in self.raw_labels]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.numeric_labels[idx]

        tensor, wavelengths_0, wavelengths_1 = parse_spx_to_split_tensor(file_path)
        wavelengths = np.stack([wavelengths_0, wavelengths_1], axis=0)

        sample = {
            'tensor': tensor,
            'wavelengths': torch.tensor(wavelengths, dtype=torch.float32),
            'label': label
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class SpectraDatasetWithMask(Dataset):
    def __init__(self, file_paths, labels=None, transform=None):
        self.file_paths = file_paths
        self.transform = transform

        if labels is None:
            raise ValueError("Необходимо передать список меток для каждого файла.")

        if len(file_paths) != len(labels):
            raise ValueError("Количество путей и меток должно совпадать.")

        self.raw_labels = [str(label).strip() for label in labels]

        # Создаем label_map
        unique_labels = sorted(set(self.raw_labels))
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}

        # Сохраняем label_map в файл
        with open('labels.json', 'w', encoding='utf-8') as f:
            json.dump(self.label_map, f, indent=2, ensure_ascii=False)

        # Числовые метки
        self.numeric_labels = [self.label_map[label] for label in self.raw_labels]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.numeric_labels[idx]

        tensor, mask = parse_spx_to_tensor_with_mask(file_path)
        sample = {'tensor': tensor, 'mask': mask, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
