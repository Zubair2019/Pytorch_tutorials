import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math

class WineDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt('/Users/mac/Downloads/wine.csv',delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = (xy.shape[0])
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.n_samples
    
dataset = WineDataset()
'''first_data = dataset[0]
features, labels = first_data
print(features, labels)'''

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True,num_workers=0)
#dataiter = iter(dataloader)
#data = next(dataiter)
#features, labels = data 
#print(features, labels)

num_epochs = 2
total_samples = len(dataset)
no_of_iterations = math.ceil(total_samples/4)
print(total_samples, no_of_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        if (i + 1)% 5 ==0:
            print(f'epoch {epoch+1}/{num_epochs} step {i+1}/{no_of_iterations} inputs {inputs.shape}')