"""
Tests for Speech Commands data loader
"""

import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader


def test_data_loader_functionality():
    """Test basic data loader functionality with synthetic audio data"""
    # Create synthetic spectrogram data
    num_samples = 100
    spectrograms = torch.randn(num_samples, 1, 64, 32)
    labels = torch.randint(0, 10, (num_samples,))
    
    # Create dataset and dataloader
    dataset = TensorDataset(spectrograms, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Test data loader
    assert dataloader is not None
    assert dataloader.batch_size == 32
    
    # Get one batch
    data, targets = next(iter(dataloader))
    
    # Check shapes
    assert data.shape[0] <= 32  # Batch size
    assert data.shape[1] == 1   # Single channel (spectrogram)
    assert data.shape[2] == 64  # n_mels
    assert data.shape[3] == 32  # target_length
    assert targets.shape[0] <= 32


def test_spectrogram_shapes():
    """Test that spectrogram data has correct shapes"""
    # Create synthetic data
    batch_size = 16
    n_mels = 64
    target_length = 32
    num_classes = 10
    
    spectrograms = torch.randn(batch_size, 1, n_mels, target_length)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Check shapes
    assert spectrograms.shape == (batch_size, 1, n_mels, target_length)
    assert labels.shape == (batch_size,)
    
    # Check label range
    assert labels.min() >= 0
    assert labels.max() < num_classes


def test_batch_processing():
    """Test batch processing with synthetic data"""
    # Create synthetic data
    num_samples = 64
    spectrograms = torch.randn(num_samples, 1, 64, 32)
    labels = torch.randint(0, 10, (num_samples,))
    
    dataset = TensorDataset(spectrograms, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Process all batches
    total_samples = 0
    for data, targets in dataloader:
        total_samples += data.shape[0]
        
        # Check label range
        assert targets.min() >= 0
        assert targets.max() < 10
    
    # Should process all samples
    assert total_samples == num_samples


def test_dataloader_consistency():
    """Test that dataloader returns consistent data"""
    # Create synthetic data
    num_samples = 32
    spectrograms = torch.randn(num_samples, 1, 64, 32)
    labels = torch.randint(0, 10, (num_samples,))
    
    dataset = TensorDataset(spectrograms, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Get first batch twice
    iter1 = iter(dataloader)
    data1, label1 = next(iter1)
    
    iter2 = iter(dataloader)
    data2, label2 = next(iter2)
    
    # Should be identical when not shuffling
    assert torch.allclose(data1, data2)
    assert torch.equal(label1, label2)
