import torch 
import torch.nn as nn
import numpy as np

from thop import profile

class NN(nn.Module):
    def create_mlp(self):

        layers = nn.ModuleList()
        ln=35
        self.inputNN = 128
        self.outputNN = 1
        
        for i in range(0, ln):
            if i == 0:
                n = self.inputNN
                m = 6144
            elif i == ln-1:
                n = 6144
                m = self.outputNN
            else:    
                n = 6144
                m = 6144

            LL = nn.Linear(int(n), int(m), bias=True)
            # initialize the weights
            # with torch.no_grad():
            # custom Xavier input, output or two-sided fill
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            # approach 1
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            layers.append(LL)
            layers.append(nn.ReLU())

        return torch.nn.Sequential(*layers)

    def prof_mlp(self):
        macs = 0
        #print(self.input)
        x = torch.empty(self.inputNN)
        macs, params = profile(self.model, inputs=(x,))
        print("macs, params: ", macs, params)

        
    def __init__ (self):
        super(NN, self).__init__()
        self.model = self.create_mlp()

        
#model = NN()
#print(model)
# x = torch.empty(16, 10)
# macs, params = profile(model, inputs=(x,))
# print(macs)
# print(params)
