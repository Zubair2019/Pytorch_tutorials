import torch
x = torch.rand(4,5)
print(x.size())
y = x.view(5,4) 
print(y.size())
z = y.view(-1,2) #-1 specify automatically given the second parameter
print(z.size())

#Pertinent to mention here that the size parameter should be same i:e the number of elements