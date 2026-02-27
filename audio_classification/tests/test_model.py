"""
Tests for Audio SNN model
"""

import pytest
import torch
from model import AudioSNN


def test_model_initialization():
    """Test that the model initializes correctly"""
    model = AudioSNN(n_mels=64, seq_length=32, num_classes=10, num_steps=25)
    
    assert model.num_steps == 25
    assert model.n_mels == 64
    assert model.seq_length == 32
    assert model.conv1.in_channels == 1
    assert model.conv1.out_channels == 32


def test_model_forward_pass():
    """Test that the model forward pass works correctly"""
    model = AudioSNN(n_mels=64, seq_length=32, num_classes=10, num_steps=10)
    batch_size = 4
    
    # Create dummy spectrogram input
    x = torch.randn(batch_size, 1, 64, 32)
    
    # Forward pass
    spk_rec, mem_rec = model(x)
    
    # Check output shapes
    assert spk_rec.shape == (10, batch_size, 10)  # (num_steps, batch_size, num_classes)
    assert mem_rec.shape == (10, batch_size, 10)
    
    # Check output types
    assert isinstance(spk_rec, torch.Tensor)
    assert isinstance(mem_rec, torch.Tensor)


def test_model_with_different_batch_sizes():
    """Test that the model works with different batch sizes"""
    model = AudioSNN(n_mels=64, seq_length=32, num_classes=10, num_steps=5)
    
    for batch_size in [1, 4, 8, 16]:
        x = torch.randn(batch_size, 1, 64, 32)
        spk_rec, mem_rec = model(x)
        
        assert spk_rec.shape == (5, batch_size, 10)
        assert mem_rec.shape == (5, batch_size, 10)


def test_model_with_different_input_sizes():
    """Test that the model adapts to different input sizes"""
    # Different n_mels and seq_length
    model = AudioSNN(n_mels=40, seq_length=64, num_classes=10, num_steps=5)
    
    x = torch.randn(4, 1, 40, 64)
    spk_rec, mem_rec = model(x)
    
    assert spk_rec.shape == (5, 4, 10)
    assert mem_rec.shape == (5, 4, 10)


def test_model_gradient_flow():
    """Test that gradients flow through the network"""
    model = AudioSNN(n_mels=64, seq_length=32, num_classes=10, num_steps=5)
    x = torch.randn(4, 1, 64, 32, requires_grad=True)
    
    spk_rec, mem_rec = model(x)
    loss = mem_rec.sum()
    loss.backward()
    
    # Check that gradients exist
    assert x.grad is not None
    assert model.conv1.weight.grad is not None
    assert model.fc1.weight.grad is not None


def test_model_cuda_compatibility():
    """Test that the model works on CUDA if available"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = AudioSNN(n_mels=64, seq_length=32, num_classes=10, num_steps=5).to(device)
        x = torch.randn(4, 1, 64, 32).to(device)
        
        spk_rec, mem_rec = model(x)
        
        assert spk_rec.device.type == 'cuda'
        assert mem_rec.device.type == 'cuda'


def test_convolutional_feature_extraction():
    """Test that convolutional layers extract features correctly"""
    model = AudioSNN(n_mels=64, seq_length=32, num_classes=10, num_steps=1)
    x = torch.randn(1, 1, 64, 32)
    
    # Test intermediate outputs by manually running through layers
    with torch.no_grad():
        # First conv + pool
        out = model.conv1(x)
        assert out.shape == (1, 32, 64, 32)  # Same size due to padding
        out = model.pool1(out)
        assert out.shape == (1, 32, 32, 16)  # Halved by pooling
        
        # Second conv + pool
        out = model.conv2(out)
        assert out.shape == (1, 64, 32, 16)
        out = model.pool2(out)
        assert out.shape == (1, 64, 16, 8)
