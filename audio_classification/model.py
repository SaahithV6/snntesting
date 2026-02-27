
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
                     
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, n_mels, seq_length)
            dummy_out = self.pool2(self.conv2(self.pool1(self.conv1(dummy_input))))
            conv_out_size = dummy_out.view(1, -1).size(1)
        

        self.fc1 = nn.Linear(conv_out_size, 256)
        self.lif3 = snn.Leaky(beta=beta)
        
        self.fc2 = nn.Linear(256, 128)
        self.lif4 = snn.Leaky(beta=beta)
        
        self.fc3 = nn.Linear(128, num_classes)
        self.lif5 = snn.Leaky(beta=beta)
        
        def forward(self, x):
            c1 = self.pool1(self.lif1(self.conv1(x))[0]) 
            c2 = self.pool2(self.lif2(self.conv2(c1))[0])
            x_flat = c2.view(c2.size(0), -1)
    
            mem3 = self.lif3.init_leaky()
            mem4 = self.lif4.init_leaky()
            mem5 = self.lif5.init_leaky()
            
            spk5_rec = []
    
            for step in range(self.num_steps):
                cur3 = self.fc1(x_flat)
                spk3, mem3 = self.lif3(cur3, mem3)
                
                cur4 = self.fc2(spk3)
                spk4, mem4 = self.lif4(cur4, mem4)
                
                cur5 = self.fc3(spk4)
                spk5, mem5 = self.lif5(cur5, mem5)
                
                spk5_rec.append(spk5)
            
            return torch.stack(spk5_rec, dim=0) 
        
        # Stack recordings
        spk_rec = torch.stack(spk5_rec, dim=0)
        mem_rec = torch.stack(mem5_rec, dim=0)
        
        return spk_rec, mem_rec
