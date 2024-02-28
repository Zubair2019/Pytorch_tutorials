import torch
x = torch.rand(5,5)
print(x)
print(x[2:5,3:5]) # the first index takes the row like 2:5 here selects rows from 2 to 4
                    # the second index the column 3:5 selects columns from 3 to 4. Largest index is greater than one
print("Thats how it is done")
print(x[2,2]) # outputs the value at 2,2 in tensor form
print(x[2,2].item()) #outputs the value at 2,2 in value form