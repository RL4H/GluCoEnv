import torch
import torch.nn as nn

n_env = 3
n_states = 4

states = torch.tensor([[1, 2, 3, 4],[5, 6, 7, 8], [9, 10, 11, 12]])
print('\nThe original matrix')
print(states)

#convert to tuple
tuple_x = torch.tensor_split(torch.flatten(states), n_env*n_states)
print('\nThe tuple of states')
print(tuple_x)

# convert tuple to matrix
x = torch.reshape(torch.tensor(tuple_x), (n_env, n_states))
x0 = torch.unsqueeze(x[:, 0], 1)
x1 = torch.unsqueeze(x[:, 1], 1)
x2 = torch.unsqueeze(x[:, 2], 1)
x3 = torch.unsqueeze(x[:, 3], 1)
print('\n converted matrix')
print(x)
print('extracted state - x0')
print(x0)

print('\n concat dxdt states back and convert to tuple')
arr = [x0, x1, x2, x3]
dxdt = torch.cat(arr, 1)
print(dxdt)
dxdt = torch.flatten(torch.reshape(dxdt, (1, n_env * n_states)))
dxdt = torch.tensor_split(dxdt, n_env * n_states)
print(dxdt)

