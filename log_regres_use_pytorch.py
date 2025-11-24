from sklearn.metrics import accuracy_score
import torch.optim as optim
from torch import nn
import pandas as pd
import numpy as np
import torch
print("\n------Logistic regression using PyTorch------\n")

print("\n------Linear layer------\n")

m = nn.Linear(5, 3)
input = torch.randn(4, 5)
output = m(input)

print('Input:', input, f'shape {input.shape}', sep='\\n')
print('\\nOutput:', output, f'shape {output.shape}', sep='\\n')

print("\n------Activation function------\n")

t = torch.randn(4)
print('Input: ', t)
print('Applying sigmoid: ', torch.sigmoid(t))
