import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import MNISTNet
from train import train_epoch, evaluate
import snntorch.functional as SF


def test_train_epoch_function():
    device = torch.device('cpu')
    model = MNISTNet(num_steps=5).to(device)
    
    num_samples = 64
    images = torch.randn(num_samples, 784)
    labels = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(images, labels)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = SF.ce_rate_loss()
    
    avg_loss, accuracy = train_epoch(model, train_loader, optimizer, loss_fn, device)
    
    assert isinstance(avg_loss, float)
    assert isinstance(accuracy, float)
    assert avg_loss > 0
    assert 0 <= accuracy <= 100


def test_test_model_function():
    device = torch.device('cpu')
    model = MNISTNet(num_steps=5).to(device)
    num_samples = 64
    images = torch.randn(num_samples, 784)
    labels = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(images, labels)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    loss_fn = SF.ce_rate_loss()
    avg_loss, accuracy = evaluate(model, test_loader, loss_fn, device)
    
    assert isinstance(avg_loss, float)
    assert isinstance(accuracy, float)
    assert avg_loss > 0
    assert 0 <= accuracy <= 100
