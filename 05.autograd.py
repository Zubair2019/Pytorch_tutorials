import torch

x = torch.randn(3, requires_grad = True) # By default is False
print("x = ",x)
y = x+2
print("y = ",y)
z = y *y *2
print("z = ",z)
'''
a = z.mean()
print("a = ",a)

a.backward()  #da/dx This runs fine without backward taking an argument because a is a scalar not tensor
print(x.grad) 
'''
v = torch.tensor([1.0,2.0,0.001],dtype=torch.float32)
z.backward(v) # z.backward() results in an error wihtout v, coz the argument can be skipped only if it is a sccalar
print("grad is ", x.grad)

'''HOW TO STOP PYTORCH FROM SAVING THE GRADS
1. x.requires_grad(False)
2. x.detach()
3. with torch.no_grad():
'''

x.requires_grad_(False)
print("new x ", x)




