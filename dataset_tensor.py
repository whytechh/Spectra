import json
import pandas as pd
from spectra_preprocessing import parse_spx_to_split_tensor, parse_spx_to_tensor_with_mask
from torch.utils.data import Dataset


class SpectraDatasetSplitChannels(Dataset):
    def __init__(self, csv_file=None, labels=None, transform=None):
        if csv_file is not None:
            df = pd.read_csv(csv_file)
            self.file_paths = df['full_path'].tolist()
            labels = df['name'].tolist()

        if labels is None:
            raise ValueError('Необходимо передать список меток для каждого файла')

        if len(self.file_paths) != len(labels):
            raise ValueError('Количество путей и меток должно совпадать')
        
        self.transform = transform

        self.raw_labels = [str(label).strip() for label in labels]

        unique_labels = sorted(set(self.raw_labels))
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}

        with open('labels_split.json', 'w', encoding='utf-8') as f:
            json.dump(self.label_map, f, indent=2, ensure_ascii=False)

        self.numeric_labels = [self.label_map[label] for label in self.raw_labels]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.numeric_labels[idx]

        tensor, _, _ = parse_spx_to_split_tensor(file_path)

        tensor = tensor.view(2, 64*64).float()

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, label


class SpectraDatasetWithMask(Dataset):
    def __init__(self, csv_file=None, labels=None, transform=None):
        df = pd.read_csv(csv_file)
        self.file_paths = df['full_path'].tolist()
        labels = df['name'].tolist()
    
        if labels is None:
            raise ValueError('Необходимо передать список меток для каждого файла')

        if len(self.file_paths) != len(labels):
            raise ValueError('Количество путей и меток должно совпадать')
        
        self.transform = transform
        self.raw_labels = [str(label).strip() for label in labels]
        unique_labels = sorted(set(self.raw_labels))
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}

        with open('labels.json', 'w', encoding='utf-8') as f:
            json.dump(self.label_map, f, indent=2, ensure_ascii=False)

        self.numeric_labels = [self.label_map[label] for label in self.raw_labels]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.numeric_labels[idx]

        tensor, mask = parse_spx_to_tensor_with_mask(file_path)

        if self.transform:
            tensor = self.transform(tensor)

        return {'tensor': tensor.float(), 'mask': mask.float(), 'label': label}
