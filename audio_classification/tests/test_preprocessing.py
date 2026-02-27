"""
Tests for audio preprocessing
"""

import pytest
import torch
from preprocessing import AudioPreprocessor, compute_mfcc


def test_audio_preprocessor_initialization():
    """Test that preprocessor initializes correctly"""
    preprocessor = AudioPreprocessor(n_mels=64, target_length=32)
    
    assert preprocessor.n_mels == 64
    assert preprocessor.target_length == 32
    assert preprocessor.sample_rate == 16000


def test_audio_preprocessing():
    """Test that audio preprocessing works correctly"""
    preprocessor = AudioPreprocessor(n_mels=64, target_length=32)
    
    # Create dummy audio (1 second at 16kHz)
    waveform = torch.randn(1, 16000)
    
    # Preprocess
    spectrogram = preprocessor(waveform)
    
    # Check shape
    assert spectrogram.shape == (1, 64, 32)
    
    # Check that output is normalized (roughly)
    assert abs(spectrogram.mean().item()) < 1.0
    assert abs(spectrogram.std().item() - 1.0) < 0.5


def test_audio_preprocessing_padding():
    """Test that short audio is padded correctly"""
    preprocessor = AudioPreprocessor(n_mels=64, target_length=32)
    
    # Create short audio
    waveform = torch.randn(1, 4000)  # 0.25 seconds
    
    spectrogram = preprocessor(waveform)
    
    # Should still be target length
    assert spectrogram.shape == (1, 64, 32)


def test_audio_preprocessing_truncation():
    """Test that long audio is truncated correctly"""
    preprocessor = AudioPreprocessor(n_mels=64, target_length=32)
    
    # Create long audio
    waveform = torch.randn(1, 32000)  # 2 seconds
    
    spectrogram = preprocessor(waveform)
    
    # Should be truncated to target length
    assert spectrogram.shape == (1, 64, 32)


def test_mfcc_computation():
    """Test MFCC computation"""
    waveform = torch.randn(1, 16000)
    
    mfcc = compute_mfcc(waveform, n_mfcc=40)
    
    # Check shape (MFCC has n_mfcc features)
    assert mfcc.shape[0] == 1  # Batch/channel
    assert mfcc.shape[1] == 40  # Number of MFCC coefficients
    
    # Check normalization
    assert abs(mfcc.mean().item()) < 1.0


def test_batch_preprocessing():
    """Test preprocessing with batch of audio"""
    preprocessor = AudioPreprocessor(n_mels=64, target_length=32)
    
    batch_size = 4
    # Create batch of audio
    waveforms = torch.randn(batch_size, 16000)
    
    # Process each in batch
    spectrograms = []
    for i in range(batch_size):
        spec = preprocessor(waveforms[i:i+1])
        spectrograms.append(spec)
    
    spectrograms = torch.cat(spectrograms, dim=0)
    
    # Check shape
    assert spectrograms.shape == (batch_size, 64, 32)


def test_preprocessor_consistency():
    """Test that same audio produces consistent output"""
    preprocessor = AudioPreprocessor(n_mels=64, target_length=32)
    
    # Create same waveform twice
    waveform1 = torch.randn(1, 16000)
    waveform2 = waveform1.clone()
    
    # Preprocess both
    spec1 = preprocessor(waveform1)
    spec2 = preprocessor(waveform2)
    
    # Should be identical
    assert torch.allclose(spec1, spec2)
