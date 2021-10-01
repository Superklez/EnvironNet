import math
import torch
from torch import Tensor
from torchaudio.transforms import ComputeDeltas, FrequencyMasking, TimeMasking,\
    MelScale, TimeStretch

sample_rate = 44100
time_stretch = TimeStretch(
    hop_length = int(sample_rate * 0.010),
    n_freq = int(sample_rate * 0.025) // 2 + 1
)
mel_scale = MelScale(
    n_mels = 64,
    sample_rate = sample_rate,
    n_stft = int(sample_rate * 0.025) // 2 + 1,
    norm = "slaney",
    mel_scale = "slaney"
)
compute_deltas = ComputeDeltas(win_length=5)

def time_shift(waveform, shift):
    if shift < 0:
        shift = abs(shift)
        temp_wave = torch.zeros(waveform.size())
        temp_wave[..., :waveform.size(-1) - shift] = waveform[..., shift:]
        temp_wave[..., waveform.size(-1) - shift:] = waveform[..., :shift] 
        waveform = temp_wave
    elif shift > 0:
        temp_wave = torch.zeros(waveform.size())
        temp_wave[..., shift:] = waveform[..., :waveform.size(-1) - shift]
        temp_wave[..., :shift] = waveform[..., waveform.size(-1) - shift:]
        waveform = temp_wave
    return waveform

freq_masking = FrequencyMasking(16)
time_masking = TimeMasking(100)

def AxisMasking(spec: Tensor, target: int):
    spec = freq_masking(spec)
    if target not in [21, 34, 39]:
        spec = time_masking(spec)
    return spec

def adjust_lengths(waveform1, waveform2):
    if waveform1.size() != waveform2.size():
        if waveform2.size(-1) < waveform1.size(-1):
            temp_wave = torch.zeros(waveform1.size())
            temp_wave[..., :waveform2.size(-1)] = waveform2
            waveform2 = temp_wave
        else:
            waveform2 = waveform2[..., :waveform1.size(-1)]
    return waveform1, waveform2

def snr_mixer(waveform1, waveform2, snr_db):
    waveform1, waveform2 = adjust_lengths(waveform1, waveform2)
    waveform1_power = waveform1.norm(p=2)
    waveform2_power = waveform2.norm(p=2)
    snr = math.exp(snr_db / 10)
    scale = snr * waveform2_power / waveform1_power
    mix_waveform = (scale * waveform1 + waveform2) / 2
    return mix_waveform