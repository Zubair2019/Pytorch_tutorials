import torch
import numpy as np

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis = 0)

x= np.array([2.0, 1.0,0.1])
output = softmax(x)
print(f'softmax numpy : {output}')

y = torch.tensor([2.0, 1.0,0.1])
output2 = torch.softmax(y, dim = 0)
print(f'softmax torch : {output2}')

 