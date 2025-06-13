import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from tqdm import tqdm

from dataset_tensor import SpectraDatasetSplitChannels
from model_1d_split import Simple1DCNN


test_csv = r'D:\projects\normalized_data\normalized_test_dataset.csv'
weights_path = r'D:\projects\project_spectra\models\model_1d_split_5.pth'
cm_top20 = r'D:\projects\project_spectra\cm_top20_split.png'


def plot_confusion_matrix(
    cm, class_names, save_path, title='Confusion Matrix', show_labels=True
):
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names if show_labels else False,
        yticklabels=class_names if show_labels else False,
    )
    plt.title(title)
    plt.ylabel('Истинные значения')
    plt.xlabel('Предсказанные значения')
    if show_labels:
        plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f'Матрица неточностей сохранена: {save_path}')


def get_top_confused_classes(cm, class_names, top_k=20):
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)

    most_confused_indices = np.argsort(cm_no_diag.sum(axis=1))[::-1][:top_k]
    top_cm = cm[np.ix_(most_confused_indices, most_confused_indices)]
    top_class_names = [class_names[i] for i in most_confused_indices]
    return top_cm, top_class_names


def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Используемое устройство: {device}')

    batch_size = 64
    test_dataset = SpectraDatasetSplitChannels(csv_file=test_csv)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

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

    precision_macro = precision_score(
        y_true, y_pred, average='macro', zero_division=0
    )
    recall_macro = recall_score(
        y_true, y_pred, average='macro', zero_division=0
    )
    f1_macro = f1_score(
        y_true, y_pred, average='macro', zero_division=0
    )

    print('\nОбщая точность:', np.mean(np.array(y_true) == np.array(y_pred)))
    print('\nМетрики качества:')
    print(f'Precision: {precision_macro:.4f}')
    print(f'Recall:    {recall_macro:.4f}')
    print(f'F1-score:  {f1_macro:.4f}')

    # Top-20 классов с наибольшими ошибками
    top_cm, top_class_names = get_top_confused_classes(cm, class_names, top_k=20)
    plot_confusion_matrix(
        top_cm,
        top_class_names,
        save_path=cm_top20,
        title='Матрица неточностей (топ-20 ошибочно определяемых классов)',
        show_labels=True,
    )


if __name__ == "__main__":
    test_model()
