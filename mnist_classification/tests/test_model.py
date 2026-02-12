import pytest
import torch
from model import MNISTNet


def test_model_initialization():
    model = MNISTNet(num_inputs=784, num_hidden=1000, num_outputs=10)
    
    assert model.num_inputs == 784
    assert model.num_hidden == 1000
    assert model.num_outputs == 10
    assert model.fc1.in_features == 784
    assert model.fc1.out_features == 1000
    assert model.fc2.in_features == 1000
    assert model.fc2.out_features == 10


def test_model_forward_pass():
    model = MNISTNet(num_steps=10)
    batch_size = 4
    
    x = torch.randn(batch_size, 784)
    spk_rec, mem_rec = model(x)
    assert spk_rec.shape == (10, batch_size, 10)  # (num_steps, batch_size, num_outputs)
    assert mem_rec.shape == (10, batch_size, 10)
    assert isinstance(spk_rec, torch.Tensor)
    assert isinstance(mem_rec, torch.Tensor)


def test_model_with_different_batch_sizes():
    model = MNISTNet(num_steps=5)
    
    for batch_size in [1, 8, 16, 32]:
        x = torch.randn(batch_size, 784)
        spk_rec, mem_rec = model(x)
        
        assert spk_rec.shape == (5, batch_size, 10)
        assert mem_rec.shape == (5, batch_size, 10)


def test_model_gradient_flow():
    model = MNISTNet(num_steps=5)
    x = torch.randn(4, 784, requires_grad=True)
    
    spk_rec, mem_rec = model(x)
    loss = mem_rec.sum()
    loss.backward()
    
    assert x.grad is not None
    assert model.fc1.weight.grad is not None
    assert model.fc2.weight.grad is not None


def test_model_cuda_compatibility():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = MNISTNet(num_steps=5).to(device)
        x = torch.randn(4, 784).to(device)
        
        spk_rec, mem_rec = model(x)
        
        assert spk_rec.device.type == 'cuda'
        assert mem_rec.device.type == 'cuda'
