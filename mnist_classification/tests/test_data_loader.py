import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms


def test_data_loader_functionality():
    num_samples = 100
    images = torch.randn(num_samples, 1, 28, 28)
    labels = torch.randint(0, 10, (num_samples,))
    
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    assert dataloader is not None
    assert dataloader.batch_size == 32
    
    data, targets = next(iter(dataloader))
    
    assert data.shape[0] <= 32  # Batch size
    assert data.shape[1] == 1   # Grayscale
    assert data.shape[2] == 28  # Height
    assert data.shape[3] == 28  # Width
    assert targets.shape[0] <= 32


def test_data_transforms():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    import numpy as np
    from PIL import Image
    
    img_array = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
    img = Image.fromarray(img_array)
    transformed = transform(img)
    assert transformed.shape == (1, 28, 28)
    assert isinstance(transformed, torch.Tensor)


def test_batch_processing():
    # Create synthetic data
    num_samples = 64
    images = torch.randn(num_samples, 1, 28, 28)
    labels = torch.randint(0, 10, (num_samples,))
    
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    total_samples = 0
    for data, targets in dataloader:
        total_samples += data.shape[0]
        
        assert targets.min() >= 0
        assert targets.max() < 10
    
    assert total_samples == num_samples
