import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from dataset_tensor import SpectraDatasetSplitChannels
from project.model_1d_split import Simple1DCNN
from sklearn.metrics import classification_report


train_csv = r'D:\projects\normalized_data\normalized_train_dataset.csv'
val_csv = r'D:\projects\normalized_data\normalized_validation_dataset.csv'
weights_path = r'D:\projects\project_spectra\models\model1d_split.pth'


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc='Train'):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Val'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def compute_metrics(model, dataloader, device, label_map):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Подсчёт метрик"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    class_names = list(label_map.keys())
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print('\nКлассификационный отчёт (по лучшей модели):')
    print(report)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Используемое устройство: {device}')

    batch_size = 64
    num_epochs = 15
    learning_rate = 1e-5
    patience = 3

    train_dataset = SpectraDatasetSplitChannels(csv_file=train_csv)
    val_dataset = SpectraDatasetSplitChannels(csv_file=val_csv)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    num_classes = len(train_dataset.label_map)

    model = Simple1DCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        print(f'\nЭпоха {epoch}/{num_epochs}')

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')

        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0 
            torch.save(model.state_dict(), weights_path)
            print('Лучшая модель сохранена')
        else:
            patience_counter += 1
            print(f'Нет улучшения ({patience_counter}/{patience})')

        if patience_counter >= patience:
            print(f'\nРанняя остановка на {epoch}-й эпохе. Лучшая эпоха: {best_epoch} с точностью {best_val_acc:.4f}')
            break
    
    model.load_state_dict(torch.load(weights_path))
    compute_metrics(model, val_loader, device, train_dataset.label_map)


if __name__ == "__main__":
    main()