"""
LIF 784 -> 1000 -> 10
"""

import torch
import torch.nn as nn
import snntorch as snn


class MNISTNet(nn.Module):

    
    def __init__(self, num_inputs=784, num_hidden=1000, num_outputs=10, 
                 beta=0.95, num_steps=25):

        super().__init__()
        
        self.num_steps = num_steps
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        
        # L1
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        
        # L2
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)
        
    def forward(self, x):

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        spk2_rec = []
        mem2_rec = []
        
        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
        
        spk_rec = torch.stack(spk2_rec, dim=0)
        mem_rec = torch.stack(mem2_rec, dim=0)
        
        return spk_rec, mem_rec
