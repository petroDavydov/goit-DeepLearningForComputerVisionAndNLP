print("\n------Model PyTorch------\n")

import torch
from torch import nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Створюємо штучні дані
X, y = make_classification(n_samples=1000, n_features=18, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Перетворюємо в тензори
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 3. Визначаємо модель
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        return out

input_dim = X_train.shape[1]
model = LogisticRegression(input_dim)

# 4. Функція втрат і оптимізатор
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 5. Навчання моделі
num_epochs = 100
for epoch in range(num_epochs):
    outputs = model(X_train).squeeze()
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 6. Тестування моделі
with torch.no_grad():
    y_pred = model(X_test).squeeze().numpy().round()
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
