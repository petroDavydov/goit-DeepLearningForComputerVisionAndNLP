import torch
import numpy as np


print(f"\n-------Single-element tensors-----\n")
x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[5, 6]])

tensor = torch.rand(3, 4)
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

print(f"\n-------In-place operations-----\n")

print(f"{tensor} \\n")
tensor.add_(5)
print(tensor)

print(f"\n---torch.tensor and numpy.array. GPU operations---\n")

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")


print("\n---change tensor---")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")


print("\n---Converting a NumPy array to a tensor---")

n = np.ones(5)
t = torch.from_numpy(n)


np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")


if torch.cuda.is_available():
    tensor = tensor.to("cuda")
    print(f"\nThis is transfer:\n {tensor}")
