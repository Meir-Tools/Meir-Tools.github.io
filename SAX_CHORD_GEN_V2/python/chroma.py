import numpy as np
import scipy.signal


def compute_chroma_scipy(y, sr=22050, nperseg=4096, noverlap=3072):
    if y is None or len(y) < 512:
        return np.zeros(12, dtype=np.float32)

    f, t, Zxx = scipy.signal.stft(
        y, fs=sr, window='hann', nperseg=nperseg, noverlap=noverlap
    )
    mag = np.mean(np.abs(Zxx), axis=1)
    mag_log = np.log1p(10.0 * mag)

    chroma = np.zeros(12, dtype=np.float32)
    for idx, freq in enumerate(f):
        if freq < 50.0 or freq > 3500.0:
            continue
        midi = 12.0 * np.log2(freq / 440.0) + 69.0
        pitch_class = int(round(midi)) % 12
        chroma[pitch_class] += mag_log[idx]

    norm = np.linalg.norm(chroma)
    if norm > 0:
        chroma /= norm
    return chroma
