import torch
import torch.nn as nn

X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = X.shape
input_size = n_features
output_size = n_features
model = nn.Linear(input_size,output_size)

learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)

print(f'Prediction before trainig: f(5) = {model(X_test).item():.3f}')

#training

for epoch in range(n_iters):
    y_pred = model(X)
    
    #loss
    l = loss(Y, y_pred)

    #gradient = backward pass
    #dw = gradient(X,Y,y_pred)
    l.backward() #dl/dw

    #update weights
    #w = w - learning_rate * dw
    optimizer.step()

    #rest gradients to zero
    optimizer.zero_grad()

    if epoch%10 ==0:
        [w,b] = model.parameters()
        print(f'epoch {epoch+1}, w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction after trainig: f(5) = {model(X_test).item():.3f}')