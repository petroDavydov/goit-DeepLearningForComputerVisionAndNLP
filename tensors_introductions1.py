import torch
import numpy as np
print(f"----------------The main sections--------------")
x = torch.empty(3, 4)
print(f"This is type(x): {type(x)}")
print(f"\nThis is torch.empty:\n {x}\n")


print(f"----------------Sections with zeros/ones/random--------------")

zeros = torch.zeros(2, 3)
print(f"This is torch.zero:\n {zeros}\n")

ones = torch.ones(2, 3)
print(f"This is torch.ones:\n {ones}\n")

torch.manual_seed(1729)
random = torch.rand(2, 3)
print(f"This is torch.rand:\n {random}\n")

print(f"----------------Sections manual seed--------------")

torch.manual_seed(1729)
random1 = torch.rand(2, 3)
print(random1)

random2 = torch.rand(2, 3)
print(random2)

torch.manual_seed(1729)
random3 = torch.rand(2, 3)
print(random3)

random4 = torch.rand(2, 3)
print(random4)

print(f"\n----------------Sections with torch.*_like-------------\n")

x = torch.empty(2, 2, 3)
print(x.shape)
print(x)

empty_like_x = torch.empty_like(x)
print(empty_like_x.shape)
print(empty_like_x)

zeros_like_x = torch.zeros_like(x)
print(zeros_like_x.shape)
print(zeros_like_x)

ones_like_x = torch.ones_like(x)
print(ones_like_x.shape)
print(ones_like_x)

rand_like_x = torch.rand_like(x)
print(rand_like_x.shape)
print(rand_like_x)

print(f"\n----------------Sections with data type-------------\n")


a = torch.ones((2, 3), dtype=torch.int16)
print(a)

b = torch.rand((2, 3), dtype=torch.float64) * 20.
print(b)

c = b.to(torch.int32)
print(c)

print(f"\n----------------Sections with attributes of tensor-------------\n")


tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


print(f"\n----------------Sections with logic & math-------------\n")

ones = torch.zeros(2, 2) + 1
twos = torch.ones(2, 2) * 2
threes = (torch.ones(2, 2) * 7 - 1) / 2
fours = twos ** 2
sqrt2s = twos ** 0.5

print(ones)
print(twos)
print(threes)
print(fours)
print(sqrt2s)


print(f"\n----------------Sections with +/-/multiplication-------------\n")


powers2 = twos ** torch.tensor([[1, 2], [3, 4]])
print(powers2)

fives = ones + fours
print(fives)

dozens = threes * fours
print(dozens)


print(f"\n----------------Sections with different form-------------\n")
print("\
a = torch.rand(2, 3)\
b = torch.rand(3, 2)\
print(a * b)\
")

print("# Answer of this section:\
Traceback (most recent call last):\
File D \
print(a * b) \
~~^~~ \
RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 1")





























