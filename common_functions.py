import os

import pandas as pd

from paths import preprocessed_data_directory, train_dataset_name, test_dataset_name


def parse_file_name(file_name):
    parts = file_name.replace('.png', '').replace('.spx', '').split('_')
    name = parts[3]
    temperature = int(parts[4].replace('T', ''))
    sample_count = int(parts[5])
    return name, temperature, sample_count


def parse_all_files_info(dataset_directory):
    items = []
    for device_path in os.listdir(dataset_directory):
        service_full_path = os.path.join(dataset_directory, device_path)
        for element_path in os.listdir(service_full_path):
            if element_path == 'All':
                continue
            element_full_path = os.path.join(service_full_path, element_path)
            for file_name in os.listdir(element_full_path):
                filename_full_path = os.path.join(element_full_path, file_name)
                name, temperature, sample_count = parse_file_name(file_name)
                items.append({
                    'device': device_path,
                    'element': element_path,
                    'file_name': file_name,
                    'full_path': filename_full_path,
                    'name': name,
                    'temperature': temperature,
                    'sample_count': sample_count
                })
    return items


def load_train_dataset():
    return pd.read_csv(os.path.join(preprocessed_data_directory, train_dataset_name), usecols=['name', 'full_path'])


def load_test_dataset():
    return pd.read_csv(os.path.join(preprocessed_data_directory, test_dataset_name), usecols=['name', 'full_path'])

def save_as_train_dataset(df):
    df.to_csv(os.path.join(preprocessed_data_directory, train_dataset_name), index=False)


def save_as_test_dataset(df):
    df.to_csv(os.path.join(preprocessed_data_directory, test_dataset_name), index=False)
