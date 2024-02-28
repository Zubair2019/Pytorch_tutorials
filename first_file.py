import torch

x = torch.rand(2,2)
y = torch.rand(2,2)
print(x)
print(y)
z = x.add_(y) #Inplace updating using trailing underscore
print(z)
print(x)