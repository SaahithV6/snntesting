
import torch
import torchaudio
import torchaudio.transforms as T


class AudioPreprocessor:

    def __init__(self, sample_rate=16000, n_mels=64, n_fft=1024, 
                 hop_length=512, target_length=32):

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.target_length = target_length
        
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
        self.amplitude_to_db = T.AmplitudeToDB()
        
    def __call__(self, waveform):

        mel_spec = self.mel_spectrogram(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)
        
        if mel_spec_db.shape[-1] < self.target_length:
            pad_amount = self.target_length - mel_spec_db.shape[-1]
            mel_spec_db = torch.nn.functional.pad(mel_spec_db, (0, pad_amount))
        elif mel_spec_db.shape[-1] > self.target_length:
            mel_spec_db = mel_spec_db[..., :self.target_length]
        
        return mel_spec_db


def compute_mfcc(waveform, sample_rate=16000, n_mfcc=40):

    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={'n_fft': 1024, 'hop_length': 512, 'n_mels': 64}
    )
    
    mfcc = mfcc_transform(waveform)
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)
    
    return mfcc
