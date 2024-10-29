import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

    
class Data_Net(nn.Module):
    def __init__(self, input_dim=28*28, out_dim=20):
        super(Data_Net, self).__init__()
        mid_num1 = 4096
        mid_num2 = 4096
        self.fc1 = nn.Linear(input_dim, mid_num1)
        self.fc2 = nn.Linear(mid_num1, mid_num2)

        self.fc3 = nn.Linear(mid_num2, out_dim, bias=False)
        nn.init.uniform_( self.fc3.weight, -1. / np.sqrt(float(input_dim)), 1. / np.sqrt(float(input_dim)) )
    def forward(self, x):
        out1 = F.relu(self.fc1(x))
        out2 = F.relu(self.fc2(out1))

        out3 = self.fc3(out2)
        norm = torch.norm(out3, p=2, dim=1, keepdim=True)
        out3 = out3 / norm
        return  out3
    

