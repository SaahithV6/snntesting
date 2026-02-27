
import torch
import torch.nn as nn
import snntorch as snn


class AudioSNN(nn.Module):

    def __init__(self, n_mels=64, seq_length=32, num_classes=10, 
                 beta=0.95, num_steps=25):

        super().__init__()
        
        self.num_steps = num_steps
        self.n_mels = n_mels
        self.seq_length = seq_length
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.lif1 = snn.Leaky(beta=beta)
        self.pool1 = nn.AvgPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.lif2 = snn.Leaky(beta=beta)
        self.pool2 = nn.AvgPool2d(2, 2)
        

        conv_out_size = 64 * (n_mels // 4) * (seq_length // 4)
        

        self.fc1 = nn.Linear(conv_out_size, 256)
        self.lif3 = snn.Leaky(beta=beta)
        
        self.fc2 = nn.Linear(256, 128)
        self.lif4 = snn.Leaky(beta=beta)
        
        self.fc3 = nn.Linear(128, num_classes)
        self.lif5 = snn.Leaky(beta=beta)
        
    def forward(self, x):

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        
        spk5_rec = []
        mem5_rec = []
        
        for step in range(self.num_steps):
            cur1 = self.conv1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1 = self.pool1(spk1)
            
            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2 = self.pool2(spk2)
            
            spk2_flat = spk2.view(spk2.size(0), -1)
            
            cur3 = self.fc1(spk2_flat)
            spk3, mem3 = self.lif3(cur3, mem3)
            
            cur4 = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)
            
            cur5 = self.fc3(spk4)
            spk5, mem5 = self.lif5(cur5, mem5)
            
            spk5_rec.append(spk5)
            mem5_rec.append(mem5)
        
        # Stack recordings
        spk_rec = torch.stack(spk5_rec, dim=0)
        mem_rec = torch.stack(mem5_rec, dim=0)
        
        return spk_rec, mem_rec
