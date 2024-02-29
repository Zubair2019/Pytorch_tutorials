import torch

#f = w * x

X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

#model prediction
def forward(x):
    return w * x

#loss = MSE
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

'''gradient descent
def gradient(x,y,y_predicted):
    return np.dot(2*x, y_predicted-y).mean()
'''



print(f'Prediction before trainig: f(5) = {forward(5):.3f}')

#trinig
learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    y_pred = forward(X)
    
    #loss
    l = loss(Y, y_pred)

    #gradient = backward pass
    #dw = gradient(X,Y,y_pred)
    l.backward() #dl/dw

    #update weights
    #w = w - learning_rate * dw
    with torch.no_grad():
        w -= learning_rate * w.grad

    #rest gradients to zero
    w.grad.zero_()

    if epoch%10 ==0:
        print(f'epoch {epoch+1}, w = {w:.3f}, loss = {l:.8f}')
        print

print(f'Prediction after trainig: f(5) = {forward(5):.3f}')