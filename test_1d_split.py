import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
from dataset_tensor import SpectraDatasetSplitChannels
from model_1d_split import Simple1DCNN


test_csv = r'D:\projects\normalized_data\normalized_test_dataset.csv'
weights_path = r'D:\projects\project_spectra\models\model_1d_split.pth'
cm_all = r'D:\projects\project_spectra\cm_all_split.png'
cm_top20 = r'D:\projects\project_spectra\cm_top20_split.png'


def plot_confusion_matrix(cm, class_names, save_path, title='Confusion Matrix'):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('Истинные значения')
    plt.xlabel('Презсказанные значения')
    plt.xticks(rotation=90)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()
    print(f'Матрица неточностей сохранена: {save_path}')


def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Используемое устройство: {device}')

    batch_size = 64
    test_dataset = SpectraDatasetSplitChannels(csv_file=test_csv)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    num_classes = len(test_dataset.label_map)
    model = Simple1DCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Вычисление предсказаний'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    class_names = list(test_dataset.label_map.keys())
    cm = confusion_matrix(y_true, y_pred)
    
    print('\nКлассификационный отчёт:')
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    plot_confusion_matrix(cm, class_names, title='Матрица неточностей (все классы)')
    plot_confusion_matrix(cm, class_names, title='Матрица неточностей (топ-20 ошибочно определяемых)', top_k=20)


if __name__ == "__main__":
    test_model()