import os
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
<<<<<<< HEAD
from efnetb0 import efnetb0
from dataset import PreprocessedDataset
=======
from model_selector import get_model
from dataset import PreprocessedDataset 
>>>>>>> e4ed3fe (update, delete and upload files)
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


def main():
    train_csv = r'preprocessed_data\train_dataset.csv'
    weights_path = r'project_spectra\models\efnetb0_weights.pth'

    batch_size = 32
    num_epochs = 1000
    learning_rate = 1e-5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True  # Ускорение сверток
    print(f'Используемое устройство: {device}')

    model_name = 'efficientnet-b0' # На выбор: efficientnet-b0, efficientnet-b3, vgg16, vgg19, resnet34, resnet50 
    model = get_model(model_name, num_classes=453, freeze=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print('Загрузка данных')
    df = pd.read_csv(train_csv, usecols=[3, 4], header=None, names=['image_path', 'label'], skiprows=1)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Создаем общий label_map
    label_map = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
    train_dataset = PreprocessedDataset(df=train_df, label_map=label_map)
    val_dataset = PreprocessedDataset(df=val_df, label_map=label_map)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Списки для метрик на обучении и валидации
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_accuracy = 0.0
    target_accuracy = 95.0

    print('Начало обучения')
    scaler = torch.amp.GradScaler()  # GradScaler без передачи device
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        train_all_labels = []
        train_all_preds = []

        # Итерация по обучающим данным
        for images, labels in train_dataloader:
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

            train_all_labels.extend(labels.cpu().numpy())
            train_all_preds.extend(predicted.cpu().numpy())

        # Вычисление средней потери и точности за эпоху
        epoch_loss = running_loss / len(train_dataloader)
        epoch_accuracy = 100 * correct / total

<<<<<<< HEAD
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
=======
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(val_dataloader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Логирование для каждой эпохи
        print(f'Эпоха [{epoch+1}/{num_epochs}] - '
            f'train_loss: {epoch_loss:.4f}, train_accuracy: {epoch_accuracy:.2f}%, '
            f'val_loss: {val_loss:.4f}, val_accuracy: {val_accuracy:.2f}%')
        
        # Сохраняем модель, если достигнута новая лучшая точность
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), weights_path)
            print(f'Новая лучшая модель сохранена с точностью {best_accuracy:.2f}%')

        # Прерывание при достижении целевой точности
        if val_accuracy >= target_accuracy:
            print(f'Достигнута целевая точность {target_accuracy}% на эпохе {epoch+1}, обучение остановлено')
            break

    # Графики
    plt.figure(figsize=(10, 4))

    # Потери
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()

    # Точность
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy per Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig('project_spectra\training_plot.png')
    
    # Вычисление Precision, Recall и F1-score
    train_precision = precision_score(train_all_labels, train_all_preds, average='weighted')
    train_recall = recall_score(train_all_labels, train_all_preds, average='weighted')
    train_f1 = f1_score(train_all_labels, train_all_preds, average='weighted')
    print(f'Итоговые метрики по обучению - Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1 Score: {train_f1:.4f}') 
>>>>>>> e4ed3fe (update, delete and upload files)


# Добавляем условие для запуска в Windows
if __name__ == '__main__':
    main()
