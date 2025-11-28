from dataclasses import dataclass

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import TargetEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import root_mean_squared_error as RMSE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from matplotlib.container import BarContainer

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
# filter warnings
warnings.filterwarnings('ignore')


print("--- Завантажимо набір даних і попередньо переглянемо:---")
df = pd.read_csv('./Module_2_Lecture_2_Class_penguins.csv')
df_sample = df.sample(5, random_state=42)

print("\n--- Виведемо базову інформацію про набір даних.---")
df.info()


print("\n--- З колонки Non-Null Count бачимо, \
що тільки декілька рядків даних мають пропущені значення.\
Можемо видалити їх з датасету.---")
df = df.dropna().reset_index(drop=True)
print(df)

print("---Подивимось на розподіл цільової змінної. Виводиться таблиця")
plt.figure(figsize=(4, 3))
ax = sns.countplot(data=df, x='species')
for i in ax.containers:
    ax.bar_label(i)
    ax.set_xlabel("value")
    ax.set_ylabel("count")

plt.suptitle("Target feature distribution")

plt.tight_layout()
plt.show()


print("--- Подивимось розподіл категоріальної змінної island. Таблиця---")
plt.figure(figsize=(4, 3))
ax = sns.countplot(data=df, x='island')
for i in ax.containers:
    ax.bar_label(i)
    ax.set_xlabel("value")
    ax.set_ylabel("count")

plt.suptitle("Island feature distribution")

plt.tight_layout()
plt.show()


print("--- Подивимось на попарний розподіл числових ознак. Таблиця ---")
# plt.figure(figsize=(6, 6))
sns.pairplot(data=df, hue='species').fig.suptitle(
    'Numeric features distribution', y=1)
plt.show()


print("--- Підготовка ознак моделі ---")

features = ['species', 'bill_length_mm',
            'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

df = df.loc[:, features]


print("--- Залишаємо усі ознаки, що є в наборі даних, та опрацьовуємо їх таким чином, щоб замість категоріальних змінних мати числові. ---")

print("--- Перетворимо категоріальну таргетну змінну в числову. ---")


df.loc[df['species'] == 'Adelie', 'species'] = 0
df.loc[df['species'] == 'Gentoo', 'species'] = 1
df.loc[df['species'] == 'Chinstrap', 'species'] = 2
df = df.apply(pd.to_numeric)

df_print = df.head(2)
print(df_print)


print("--- Представимо матрицю ознак X та вектор nаргетової змінної yy як numpy-масив. ---")


X = df.drop('species', axis=1).values
y = df['species'].values

print(X)
print(y)


print("--- Щоб гарантувати, що ознаки будуть представлені в одному масштабі, використаємо StandardScaler ---")


scaler = StandardScaler()
X = scaler.fit_transform(X)

print(X)


print("--- Розділимо дані на тренувальні та тестові.---")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.33, stratify=y)


print("--- Для подальшої роботи з інструментами фреймворку PyTorch перетворимо дані з numpy-масивів у torch.tensor.---")


X_train = torch.Tensor(X_train).float()
y_train = torch.Tensor(y_train).long()

X_test = torch.Tensor(X_test).float()
y_test = torch.Tensor(y_test).long()

print(X_train[:1])
print(y_train[:10])


# Виведення
# tensor([[ 1.2650,  0.9842, -0.3549, -0.8172]])
# tensor([2, 0, 1, 1, 1, 2, 0, 1, 1, 0])


print("--- Задача багатокласової класифікації. Моделювання та аналіз результатів---")


class LinearModel(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=20, out_dim=3):
        super().__init__()

        self.features = torch.nn.Sequential(

            nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),

            nn.Linear(hidden_dim, out_dim),
            # nn.Softmax()
        )

    def forward(self, x):
        output = self.features(x)
        return output


print("--- Ініціалізуємо модель ---")

model = LinearModel(X_train.shape[1], 20, 3)

print("--- Тренування/Тестування/Графіки ---")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

num_epoch = 400

train_loss = []
test_loss = []

train_accs = []
test_accs = []

model.eval()

for epoch in range(num_epoch):

    # train the model
    model.train()

    outputs = model(X_train)

    loss = criterion(outputs, y_train)
    train_loss.append(loss.cpu().detach().numpy())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    acc = 100 * torch.sum(y_train == torch.max(outputs.data, 1)
                          [1]).double() / len(y_train)
    train_accs.append(acc)

    if (epoch+1) % 10 == 0:
        print('Epoch [%d/%d] Loss: %.4f   Acc: %.4f'
              % (epoch+1, num_epoch, loss.item(), acc.item()))

    # test the model

    model.eval()
    with torch.no_grad():
        outputs = model(X_test)

        loss = criterion(outputs, y_test)
        test_loss.append(loss.cpu().detach().numpy())

        acc = 100 * \
            torch.sum(y_test == torch.max(outputs.data, 1)
                      [1]).double() / len(y_test)
        test_accs.append(acc)


print("--- Графік зміни функції втрат під час тренування та тестування ---")

plt.figure(figsize=(4, 3))
plt.plot(train_loss, label='Train')
plt.plot(test_loss, label='Validation')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy Loss')
plt.title('Training vs Validation Loss')
plt.show()


print("--- Графік зміни точності під час тренування та тестування ---")

plt.figure(figsize=(4, 3))
plt.plot(train_accs, label='Train')
plt.plot(test_accs, label='Validation')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training vs Validation Accuracy')
plt.show()


print("--- Остаточна точність на тестовому наборі даних ---")

model.eval()
with torch.no_grad():
    outputs = model(X_test)

    acc = 100 * torch.sum(y_test == torch.max(outputs.data, 1)
                          [1]).double() / len(y_test)
    print(f'Test Accuracy: {acc:.4f} %')
