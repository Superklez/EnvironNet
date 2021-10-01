import numpy as np
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt

def num_params(
    model: object
) -> int:
    return sum([p.numel() for p in model.parameters() if p.requires_grad])

def plot_spectrogram(
    spectrogram: np.ndarray,
    sample_rate: int,
    hop_length: int = 0.010,
    title: str = 'Spectrogram (db)'
):
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    im = librosa.display.specshow(
        spectrogram,
        x_axis = 'time',
        y_axis = 'mel',
        sr = sample_rate,
        hop_length = int(sample_rate * hop_length),
        fmin = 0,
        fmax = sample_rate // 2,
        cmap = 'viridis',
        ax = ax
    )
    ax.set_title(title)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    fig.colorbar(im, ax=ax)
    plt.show(block=False)
    
def plot_waveform(waveform, sample_rate=44100, title="Waveform", **kwargs):
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    librosa.display.waveplot(waveform, sample_rate, ax=ax, **kwargs)
    ax.set_ylabel("Amplitude")
    fig.suptitle(title)
    plt.show(block=False)

def play_audio(waveform, sample_rate=44100):
    ipd.display(ipd.Audio(waveform, rate=sample_rate))