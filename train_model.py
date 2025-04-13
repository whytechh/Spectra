import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from efnetb0 import efnetb0
from dataset import PreprocessedDataset
from sklearn.metrics import precision_score, recall_score, f1_score


def main():
    train_csv = r'preprocessed_data\train_dataset.csv'
    weights_path = r'project_spectra\models\efnetb0_weights.pth'

    batch_size = 32
    num_epochs = 100
    learning_rate = 1e-5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True  # Ускорение сверток
    print(f'Используемое устройство: {device}')

    model = efnetb0()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print('Загрузка данных')
    dataset_train = PreprocessedDataset(csv_file=train_csv)
    # Установите num_workers=0, чтобы избежать ошибок с многозадачностью
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Списки для метрик за все эпохи
    all_labels = []
    all_preds = []

    print('Начало обучения')
    scaler = torch.amp.GradScaler()  # GradScaler без передачи device
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        # Итерация по обучающим данным
        for images, labels in dataloader_train:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Используем AMP для ускорения
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Обратный проход
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            # Подсчет точности
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Собираем все метки и предсказания для дальнейшего вычисления метрик
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

        # Вычисление средней потери и точности за эпоху
        epoch_loss = running_loss / len(dataloader_train)
        epoch_accuracy = 100 * correct / total

        # Логирование результатов
        print(f"Эпоха [{epoch + 1}/{num_epochs}] - Лосс: {epoch_loss:.4f}, Точность: {epoch_accuracy:.2f}%")

    # Вычисление Precision, Recall и F1-score
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Итоговые метрики - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    print(f'Сохранение весов модели: {weights_path}')


# Добавляем условие для запуска в Windows
if __name__ == '__main__':
    main()
