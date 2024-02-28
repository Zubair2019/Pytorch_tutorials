import torch 
import numpy

a = torch.ones(5)
print(type(a))
print(a)
x = a.numpy()
print(type(x))
print(x)

a.add_(2)

print(a)
print(x)  # Since both the numpy array and the tensor are on CPU so a change to one of them results in 
            #A CHANGE TO THE OTHER AS WELL

b = numpy.ones(6)
print(b, type(b))

y = torch.from_numpy(b)
print(y, type(y))

z = y.add_(1)
print(z,b) # Similar phenomena but in opposite direction