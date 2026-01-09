# topic_14_optimization.py
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
import os

# -----------------------
# Конфігурація та допоміжні функції
# -----------------------


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_data(root='./data', batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_dataset = datasets.MNIST(
        root=root, train=True, download=True, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size])

    test_dataset = datasets.MNIST(
        root=root, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def visualize_training_history(train_losses, train_accs, val_losses, val_accs, pause_seconds=5):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Train Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.pause(pause_seconds)
    plt.close()

# -----------------------
# Моделі
# -----------------------


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        # очікуємо x у формі [batch, 1, 28, 28] або [batch, 784]
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class NetWithDropout(nn.Module):
    def __init__(self):
        super(NetWithDropout, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)


class NetWithBN(nn.Module):
    def __init__(self):
        super(NetWithBN, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)

# Модель для гіпероптимізації (з можливістю dropout і batchnorm)


class NetForOpt(nn.Module):
    def __init__(self, dropout_rate=0.2, use_batchnorm=False):
        super(NetForOpt, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(dropout_rate)
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(512)

    def forward(self, x):
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        x = self.fc1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return self.fc3(x)

# -----------------------
# Основні блоки тренування (збережені як у оригіналі, але виправлені)
# -----------------------


def train_no_regularization(device, train_loader, val_loader, num_epochs=20):
    print("ВИЗНАЧАЄМО ДЕВАЙС НА ЯКОМУ ТРЕНУЄМОСЯ")
    print(device)
    print("ТРЕНУВАННЯ МОДЕЛІ БЕЗ РЕГУЛЯЦІЇ")

    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = 100. * train_correct / train_total

        train_losses.append(train_loss)
        train_accs.append(train_accuracy)

        # validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100. * val_correct / val_total

        val_losses.append(val_loss)
        val_accs.append(val_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(
            f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
        print('-' * 50)

    print("Вивід графіків для моделі без регуляції")
    visualize_training_history(train_losses, train_accs, val_losses, val_accs)


def train_with_l2(device, train_loader, val_loader, num_epochs=20, weight_decay=1e-3):
    print("Тренування і графіки для моделі з регуляцією")
    print("ВИЗНАЧАЄМО ДЕВАЙС НА ЯКОМУ ТРЕНУЄМОСЯ з регуляцією")
    print(device)

    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = 100. * train_correct / train_total

        train_losses.append(train_loss)
        train_accs.append(train_accuracy)

        # validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100. * val_correct / val_total

        val_losses.append(val_loss)
        val_accs.append(val_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(
            f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
        print('-' * 50)

    visualize_training_history(train_losses, train_accs, val_losses, val_accs)


def train_with_early_stopping(device, train_loader, val_loader, num_epochs=20, patience=2):
    print("Early stopping training")
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    best_val_loss = float('inf')
    counter = 0

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    visualize_training_history(train_losses, train_accs, val_losses, val_accs)


def train_with_dropout(device, train_loader, val_loader, num_epochs=20):
    print("ВИЗНАЧАЄМО ДЕВАЙС НА ЯКОМУ ТРЕНУЄМОСЯ з DROPOUT")
    print(device)

    model = NetWithDropout().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    visualize_training_history(train_losses, train_accs, val_losses, val_accs)


def train_with_batchnorm(device, train_loader, val_loader, num_epochs=20):
    print("ВИЗНАЧАЄМО ДЕВАЙС НА ЯКОМУ ТРЕНУЄМОСЯ з BatchNorm")
    print(device)

    model = NetWithBN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    visualize_training_history(train_losses, train_accs, val_losses, val_accs)

# -----------------------
# Optuna objective (виправлено: device, flatten, prev_loss, DataLoader)
# -----------------------


def objective(trial, device, train_dataset, test_dataset):
    # determine hyperparameters for optimization
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    use_batchnorm = trial.suggest_categorical('use_batchnorm', [True, False])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

    # create a model with selected hyperparameters
    model = NetForOpt(dropout_rate=dropout_rate,
                      use_batchnorm=use_batchnorm).to(device)

    # training settings
    train_loader_local = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    prev_loss = float('inf')
    model.train()
    for epoch in range(10):  # limit to 10 epochs for speed
        for batch_idx, (data, target) in enumerate(train_loader_local):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # Early stopping heuristic inside objective
        if loss.item() > prev_loss:
            break
        prev_loss = loss.item()

    # model evaluation on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        test_loader_local = DataLoader(
            test_dataset, batch_size=1000, shuffle=False)
        for data, target in test_loader_local:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / total
    return accuracy

# -----------------------
# Optuna visualization helper (matplotlib)
# -----------------------


def optuna_plots(study, pause_seconds=5):
    try:
        import optuna.visualization.matplotlib as ovm
    except Exception:
        print("Optuna visualization (matplotlib) not available.")
        return

    plt.figure(figsize=(10, 6))
    ovm.plot_optimization_history(study)
    plt.title("Optimization History")
    plt.tight_layout()
    plt.pause(pause_seconds)
    plt.close()

    plt.figure(figsize=(10, 6))
    ovm.plot_param_importances(study)
    plt.title("Parameter Importances")
    plt.tight_layout()
    plt.pause(pause_seconds)
    plt.close()

    plt.figure(figsize=(10, 6))
    ovm.plot_parallel_coordinate(study)
    plt.title("Parallel Coordinate Plot")
    plt.tight_layout()
    plt.pause(pause_seconds)
    plt.close()

    plt.figure(figsize=(10, 6))
    ovm.plot_slice(study)
    plt.title("Slice Plot")
    plt.tight_layout()
    plt.pause(pause_seconds)
    plt.close()

# -----------------------
# Головна частина: if __name__ == "__main__"
# -----------------------


if __name__ == "__main__":
    print("ПОЧАТОК РОБОТИ КОДУ")

    # Підготовка даних
    print("ПОЧАТОК ЗАГРУЗКИ ДАНИХ")
    batch_size = 64
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = prepare_data(
        batch_size=batch_size)

    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    print(f"Number of batches in train_loader: {len(train_loader)}")

    device = get_device()
    print("ВИЗНАЧАЄМО ДЕВАЙС НА ЯКОМУ ТРЕНУЄМОСЯ")
    print(device)

    # Блок -5- Без Регуляції
    train_no_regularization(device, train_loader, val_loader, num_epochs=20)

    # Блок -7- З регуляцією (L2)
    train_with_l2(device, train_loader, val_loader,
                  num_epochs=20, weight_decay=1e-3)

    # Блок -8- Early stopping
    train_with_early_stopping(device, train_loader,
                              val_loader, num_epochs=20, patience=2)

    # Блок -10- Dropout
    train_with_dropout(device, train_loader, val_loader, num_epochs=20)

    # Блок -13- BatchNorm
    train_with_batchnorm(device, train_loader, val_loader, num_epochs=20)

    # Блок -14/15/16 - Hyperparameter optimization with Optuna
    print("Починаємо оптимізацію гіперпараметрів з Optuna (може зайняти час)...")
    study = optuna.create_study(direction='maximize')
    # Optuna objective wrapper to pass device and datasets
    study.optimize(lambda trial: objective(
        trial, device, train_dataset, test_dataset), n_trials=20)

    print('Number of finished trials:', len(study.trials))
    print('Best trial:')
    trial = study.best_trial
    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    # Optuna plots (matplotlib)
    optuna_plots(study)

    print("Усі блоки виконано.")
