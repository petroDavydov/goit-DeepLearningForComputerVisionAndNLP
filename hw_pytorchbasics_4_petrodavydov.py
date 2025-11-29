from dataclasses import dataclass

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# 2. Підготовка даних

df = pd.read_csv('./ConcreteStrengthData.csv')
print(df)  # вивід df

df_info = df.info()
print(df_info)

# 2. Підготовкка даних

X = df.drop(columns=['Strength'])   # ознаки
y = df['Strength'].values           # цільова змінна

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# нормалізація
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# перетворення у тензори
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Dataset і DataLoader
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Початкові базові експеременти

# 3. Створення моделі


class ConcreteNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)   # вихідний шар для регресії
        )

    def forward(self, x):
        return self.net(x)


# 4. Налаштування навчання
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConcreteNet(in_dim=X_train.shape[1]).to(device)
criterion = nn.MSELoss()   # функція втрат для регресії
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
num_epochs = 100

# 5. Навчання моделі
train_losses = []
for epoch in range(1, num_epochs+1):
    model.train()
    epoch_losses = []
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    train_losses.append(np.mean(epoch_losses))

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            preds, trues = [], []
            for Xb, yb in test_loader:
                Xb = Xb.to(device)
                out = model(Xb)
                # заміна flatten на reshape(-1)
                preds.extend(out.cpu().numpy().reshape(-1))
                trues.extend(yb.numpy().reshape(-1))          # те саме для yb
            preds, trues = np.array(preds), np.array(trues)

            # перевірка на NaN/Inf
            if np.isnan(preds).any() or np.isnan(trues).any():
                print("⚠️ Warning: NaN detected in predictions or targets")
            elif np.isinf(preds).any() or np.isinf(trues).any():
                print("⚠️ Warning: Inf detected in predictions or targets")
            else:
                rmse = np.sqrt(mean_squared_error(trues, preds))
                print(
                    f'Epoch [{epoch}/{num_epochs}], Loss: {train_losses[-1]:.4f}, RMSE: {rmse:.6f}')

# 6. Оцінка моделі
model.eval()
with torch.no_grad():
    y_pred = model(X_test_t.to(device)).cpu().numpy().flatten()
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('MSE:', mse)
print('MAE:', mae)
print('R2:', r2)

# 7. Аналіз результатів
print()
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Strength')
plt.ylabel('Predicted Strength')
plt.title('Actual vs Predicted')
plt.show()
print()
plt.figure()
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Train Loss (MSE)')
plt.title('Training Loss Curve')
plt.show()

# Варіант максимальної оптимізації, для експерименту

# 8. Оптимізація моделі (приклад альтернативної архітектури)


class ConcreteNetDeep(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        return self.net(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConcreteNetDeep(in_dim=X_train.shape[1]).to(device)
criterion = nn.MSELoss()   # функція втрат для регресії
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
num_epochs = 400

# Навчання моделі
train_losses = []
for epoch in range(1, num_epochs+1):
    model.train()
    epoch_losses = []
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    train_losses.append(np.mean(epoch_losses))

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            preds, trues = [], []
            for Xb, yb in test_loader:
                Xb = Xb.to(device)
                out = model(Xb)
                preds.extend(out.cpu().numpy().flatten())
                trues.extend(yb.numpy().flatten())
            preds, trues = np.array(preds), np.array(trues)
            rmse = np.sqrt(mean_squared_error(trues, preds))
        print(
            f'Epoch [{epoch}/{num_epochs}], Loss: {train_losses[-1]:.4f}, RMSE: {rmse:.6f}')

# Оцінка моделі
model.eval()
with torch.no_grad():
    y_pred = model(X_test_t.to(device)).cpu().numpy().flatten()
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('MSE:', mse)
print('MAE:', mae)
print('R2:', r2)

#  Аналіз результатів
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Strength')
plt.ylabel('Predicted Strength')
plt.title('Actual vs Predicted')
plt.show()
print()
plt.figure()
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Train Loss (MSE)')
plt.title('Training Loss Curve')
plt.show()
