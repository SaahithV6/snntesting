import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from preprocessing import AudioPreprocessor


class SpeechCommandsDataset(Dataset):
    def __init__(self, root, subset='training', download=True, 
                 preprocessor=None, target_classes=None):

        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root, 
            download=download,
            subset=subset
        )
        
        self.preprocessor = preprocessor or AudioPreprocessor()
        
        all_labels = set()
        for i in range(len(self.dataset)):
            _, _, label, *_ = self.dataset[i]
            all_labels.add(label)
        
        if target_classes is None:
            target_classes = sorted(list(all_labels))[:10]
        
        self.target_classes = target_classes
        self.label_to_idx = {label: idx for idx, label in enumerate(target_classes)}
        
        self.valid_indices = []
        for i in range(len(self.dataset)):
            _, _, label, *_ = self.dataset[i]
            if label in self.label_to_idx:
                self.valid_indices.append(i)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):

        real_idx = self.valid_indices[idx]
        waveform, sample_rate, label, *_ = self.dataset[real_idx]
        spectrogram = self.preprocessor(waveform)
        label_idx = self.label_to_idx[label]
        
        return spectrogram, label_idx
    
    def get_classes(self):
        return self.target_classes


def get_speech_commands_loaders(batch_size=32, data_dir='./data', 
                                num_workers=0, n_mels=64, target_length=32):
    preprocessor = AudioPreprocessor(n_mels=n_mels, target_length=target_length)

    train_dataset = SpeechCommandsDataset(
        data_dir,
        subset='training',
        download=True,
        preprocessor=preprocessor
    )
    
    test_dataset = SpeechCommandsDataset(
        data_dir,
        subset='testing',
        download=True,
        preprocessor=preprocessor,
        target_classes=train_dataset.get_classes()  # Use same classes
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset.get_classes()
