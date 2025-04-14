import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from dataset import PreprocessedDataset
from torch.utils.data import DataLoader
from model_selector import get_model
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Пути и гиперпараметры
weights_path = r'project_spectra\models\efnetb0_weights.pth'
test_csv = r'preprocessed_data\test_dataset.csv'

batch_size = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Инициализация модели
print('Загрузка модели')
model_name = 'efficientnet-b0' # На выбор: efficientnet-b0, efficientnet-b3, vgg16, vgg19, resnet34, resnet50 
model = get_model(model_name, num_classes=453, freeze=True)
model.to(device)

# Данные для тестирования
print('Загрузка данных')
dataset_test = PreprocessedDataset(csv_file=test_csv)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size)

# Тестирование модели
print('Начало тестирования')
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in dataloader_test:
        images, labels = images.to(device), labels.to(device)

        # Прогон модели
        outputs = model(images)

        # Получаем предсказания
        _, predicted = torch.max(outputs, 1)

        # Собираем все метки и предсказания
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# Вычисление метрик
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

# Вывод результатов
print(f'Итоговые метрики на тестовом наборе:')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Вычисление точности
accuracy = 100 * (torch.tensor(all_preds) == torch.tensor(all_labels)).sum().item() / len(all_labels)
print(f'Точность: {accuracy:.2f}%')

# Матрица неточностей
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('project_spectra\confusion_matrix.png')

# Сохранение файла 
results_df = pd.DataFrame({
    'actual': all_labels,
    'predicted': all_preds
})
results_df.to_csv('project_spectra\test_predictions.csv', index=False)