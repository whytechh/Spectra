import pandas as pd
import json
from collections import defaultdict, deque
import os

# Загружаем json файл с синонимами и создаем связи между ними
def load_synonyms(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        synonyms = json.load(f)

    connections = defaultdict(set)
    for main_name, similar_names in synonyms.items():
        for name in similar_names:
            connections[main_name.lower()].add(name.lower())
            connections[name.lower()].add(main_name.lower())

    return connections


# Объединяем связанные имена в группы (синоним - эталон)
def name_mapping(connections):
    already_checked = set()
    groups = []

    for name in connections:
        if name not in already_checked:
            group = []
            queue = deque([name])
            while queue:
                current = queue.popleft()
                if current not in already_checked:
                    already_checked.add(current)
                    group.append(current)
                    queue.extend(connections[current] - already_checked)
            groups.append(sorted(group))

    name_map = {}
    for group in groups:
        main_name = group[0]
        for name in group:
            name_map[name] = main_name
    return name_map


# Применяем изменения к csv-файлу и сохраняем 
def normalize_csv(csv_path, name_mapping, output_folder):
    df = pd.read_csv(csv_path)
    df['name'] = df['name'].map(lambda x: name_mapping.get(str(x).strip().lower(), x))

    original_name = os.path.basename(csv_path)
    output_name = f'normalized_{original_name}'
    output_path = os.path.join(output_folder, output_name)

    df.to_csv(output_path, index=False)
    print(f'Сохранено: {output_path}')


if __name__ == '__main__':

    json_path = r'Spectra\EtalonsEquivalents.json'
    input_dir = r'Spectra\preprocessed_data'

    output_dir = r'Spectra\normalized_data'
    os.makedirs(output_dir, exist_ok=True)

    filenames = ['test_dataset.csv', 'train_dataset.csv', 'validation_dataset.csv']
    csv_files = [os.path.join(input_dir, name) for name in filenames]

    connections = load_synonyms(json_path)
    name_map = name_mapping(connections)

    for csv_path in csv_files:
        normalize_csv(csv_path, name_map, output_dir)
