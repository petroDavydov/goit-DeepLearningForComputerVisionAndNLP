import torch
import numpy as np

print(f"\n----------Indexing, slicing, merging, and translating tensors-------------------\n")


tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:, 1] = 0
print(tensor)

print(f"\n----------United tensors  /cat/stack-------------------\n")

x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[5, 6]])
# Конкатенація вздовж 0-го виміру
result = torch.cat((x, y), dim=0)
print(result)

print("\n------'stack' waiting 2 arguments--------------\n")

# x = torch.tensor([[1, 2], [3, 4]])
# y = torch.tensor([[5, 6]])
# # Стек вздовж нового виміру (вимір 0)
# result = torch.stack((x, y), dim=0)
# print(result)


print("\n------tensor broadcasting--------------\n")

rand = torch.rand(2, 4)
doubled = rand * (torch.ones(1, 4) * 2)

print(rand)
print(doubled)

print("\n------rules of broadcasting----------\n")

a = torch.ones(4, 3, 2)

b = a * torch.rand(   3, 2) # 3rd & 2nd dims identical to a, dim 1 absent
print(b)

c = a * torch.rand(   3, 1) # 3rd dim = 1, 2nd dim identical to a
print(c)

d = a * torch.rand(   1, 2) # 3rd dim identical to a, 2nd dim = 1
print(d)























