import os
import random
import pandas as pd

import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from augmentations import AxisMasking, snr_mixer, time_shift, time_stretch, \
    mel_scale, sample_rate

class AudioDataset(Dataset):
    def __init__(self,
        annotations: pd.DataFrame,
        root_dir: str,
        training: bool = False,
        shift_range: list = [0, 1],
        snr_range: list = [0, 10],
        stretch_range: list = [0.8, 1.2],
    ):
        """
        Inputs:
        ---------
            csv_file (str): Path to the csv file with annotations.
            root_dir (str): Directory to audio data.
        """
        self.annotations = annotations
        self.root_dir = root_dir
        self.training = training
        self.shift_range = shift_range
        self.snr_range = snr_range
        self.stretch_range = stretch_range

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_path = os.path.join(self.root_dir,
            self.annotations.loc[index, "filename"])
        wave1, sr = torchaudio.load(audio_path)
        target = self.annotations.loc[index, "target"]

        if self.training:
            shift1 = int(random.uniform(self.shift_range[0],
                self.shift_range[1]) * sr)
            wave1 = time_shift(wave1, shift1)
            mix_annot = self.annotations[self.annotations["target"] != \
                target].sample()
            wave2, _ = torchaudio.load(
                os.path.join(
                    self.root_dir,
                    mix_annot["filename"].values[0])
            )
            target2 = mix_annot["target"].values[0]
            shift2 = int(random.uniform(self.shift_range[0],
                self.shift_range[1]) * sr)
            wave2 = time_shift(wave2, shift2)
            snr = int(random.uniform(self.snr_range[0], self.snr_range[1]))
            wave1 = snr_mixer(wave1, wave2, snr)

        spec = torch.stft(
            wave1,
            n_fft = int(sample_rate * 0.025),
            hop_length = int(sample_rate * 0.010),
            window = torch.hann_window(int(sample_rate * 0.025)),
            return_complex = True
        )

        if self.training:
            ts = round(random.uniform(self.stretch_range[0],
                self.stretch_range[1]), 2)
            if (ts <= 0.95) or (ts >= 1.05):
                spec = time_stretch(spec, ts)
        
        spec = torch.square(torch.abs(spec))
        spec = mel_scale(spec)
        spec = torch.log(spec + 1e-9)
        # delta = compute_deltas(spec)
        # spec = torch.cat([spec, delta], dim=0)
        spec = (spec - spec.mean([1, 2], keepdim=True)) / spec.std([1, 2],
            keepdim=True)
        if self.training:
            spec = AxisMasking(spec, target)
        
        return wave1, spec, target

def pad_sequence(batch: Tensor):
    batch = [item.permute(2, 0, 1) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True,
        padding_value=0)
    return batch.permute(0, 2, 3, 1)

def collate_fn(batch: Tensor):
    batch = sorted(batch, key=lambda x: x[1].shape[-1], reverse=True)
    _, tensors, targets = zip(*batch)
    tensors = pad_sequence(tensors)
    targets = torch.tensor(targets)
    return tensors, targets