import threading
import time
import numpy as np
import scipy.signal
import soundcard as sc
from typing import Optional, List, Dict, Callable

from chord_detector import ChordDetector, transpose_for_alto_sax

def compute_chroma_scipy(y: np.ndarray, sr: int = 22050, nperseg: int = 4096, noverlap: int = 3072) -> np.ndarray:
    """
    Computes a 12-bin pitch class chromagram using pure NumPy and SciPy STFT.
    No Numba / Librosa DLL dependencies required!
    """
    if y is None or len(y) < 512:
        return np.zeros(12, dtype=np.float32)

    # Compute STFT
    f, t, Zxx = scipy.signal.stft(
        y, fs=sr, window='hann', nperseg=nperseg, noverlap=noverlap
    )
    
    # Average magnitude across time frames
    mag = np.mean(np.abs(Zxx), axis=1)

    # Log compression for audio dynamics
    mag_log = np.log1p(10.0 * mag)

    chroma = np.zeros(12, dtype=np.float32)
    for idx, freq in enumerate(f):
        if freq < 50.0 or freq > 3500.0:
            continue
        
        # MIDI pitch number
        midi = 12.0 * np.log2(freq / 440.0) + 69.0
        pitch_class = int(round(midi)) % 12
        chroma[pitch_class] += mag_log[idx]

    # Normalize chroma vector
    norm = np.linalg.norm(chroma)
    if norm > 0:
        chroma /= norm

    return chroma


class AudioEngine:
    def __init__(
        self,
        sample_rate: int = 22050,
        buffer_seconds: float = 1.5,
        hop_seconds: float = 0.15,
        smoothing_alpha: float = 0.35
    ):
        self.sample_rate = sample_rate
        self.buffer_seconds = buffer_seconds
        self.hop_seconds = hop_seconds
        self.smoothing_alpha = smoothing_alpha

        self.buffer_size = int(self.sample_rate * self.buffer_seconds)

        self.audio_ring = np.zeros(self.buffer_size, dtype=np.float32)
        self.audio_lock = threading.Lock()

        self.is_running = False
        self.selected_device_id: Optional[str] = None
        self.selected_is_loopback: bool = True  # Default to system speakers loopback!

        self.chord_detector = ChordDetector(energy_threshold=0.02)

        # Smoothed chroma history vector (12,)
        self.smoothed_chroma = np.zeros(12, dtype=np.float32)

        # Callback for UI updates
        self.on_chord_detected: Optional[Callable[[str, str, float, np.ndarray, float], None]] = None

        self.worker_thread: Optional[threading.Thread] = None

        # Auto-detect default loopback device ID on init
        self._init_default_device()

    def _init_default_device(self):
        """Sets default device to system speakers loopback if available."""
        try:
            default_spk = sc.default_speaker()
            if default_spk:
                self.selected_device_id = default_spk.id
                self.selected_is_loopback = True
        except Exception:
            devices = self.get_input_devices()
            if devices:
                self.selected_device_id = devices[0]['id']
                self.selected_is_loopback = devices[0]['is_loopback']

    @staticmethod
    def get_input_devices() -> List[Dict[str, any]]:
        """
        Returns a list of all audio input devices (System Speaker Loopbacks + Physical Mics).
        """
        devices = []
        try:
            default_spk_id = None
            try:
                default_spk_id = sc.default_speaker().id
            except Exception:
                pass

            all_mics = sc.all_microphones(include_loopback=True)
            for idx, mic in enumerate(all_mics):
                is_loopback = getattr(mic, 'isloopback', False)
                is_default_spk = (default_spk_id is not None and mic.id == default_spk_id and is_loopback)
                
                prefix = "🔊 [שמע מערכת / רמקולים]" if is_loopback else "🎤 [מיקרופון]"
                if is_default_spk:
                    prefix = "🔊★ [ברירת מחדל - רמקולים]"

                name_display = f"{prefix} {mic.name}"
                devices.append({
                    'index': idx,
                    'id': mic.id,
                    'name': mic.name,
                    'display_name': name_display,
                    'is_loopback': is_loopback,
                    'is_default': is_default_spk
                })

            # Sort so system speakers loopback comes FIRST
            devices.sort(key=lambda d: (not d['is_default'], not d['is_loopback'], d['name']))

        except Exception as e:
            print(f"Error querying soundcard devices: {e}")

        return devices

    def set_device(self, device_id: str, is_loopback: bool = True):
        """Set active audio capture device."""
        self.selected_device_id = device_id
        self.selected_is_loopback = is_loopback
        if self.is_running:
            self.stop()
            self.start()

    def start(self):
        """Starts audio recording stream and processing loop."""
        if self.is_running:
            return

        self.is_running = True
        self.worker_thread = threading.Thread(target=self._recording_and_processing_loop, daemon=True)
        self.worker_thread.start()

    def stop(self):
        """Stops audio stream and processing loop."""
        self.is_running = False
        if self.worker_thread is not None and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.5)
            self.worker_thread = None

    def _recording_and_processing_loop(self):
        """Background thread recording audio from soundcard loopback/mic and running SciPy chromagram extraction."""
        target_mic = None

        try:
            all_mics = sc.all_microphones(include_loopback=True)
            # Find matching device
            if self.selected_device_id:
                for mic in all_mics:
                    if mic.id == self.selected_device_id:
                        target_mic = mic
                        break
            
            if target_mic is None:
                try:
                    default_spk = sc.default_speaker()
                    target_mic = sc.get_microphone(id=default_spk.name, include_loopback=True)
                except Exception:
                    if all_mics:
                        target_mic = all_mics[0]

            if target_mic is None:
                print("No suitable audio input or loopback device found.")
                self.is_running = False
                return

            print(f"Starting audio capture from: {target_mic.name} (Loopback: {getattr(target_mic, 'isloopback', False)})")

            chunk_samples = int(self.sample_rate * self.hop_seconds)

            with target_mic.recorder(samplerate=self.sample_rate, channels=1) as recorder:
                while self.is_running:
                    chunk = recorder.record(numframes=chunk_samples)
                    if chunk is None or len(chunk) == 0:
                        time.sleep(0.05)
                        continue

                    mono = chunk.flatten().astype(np.float32)
                    n_samples = len(mono)

                    # Update ring buffer
                    with self.audio_lock:
                        if n_samples >= self.buffer_size:
                            self.audio_ring[:] = mono[-self.buffer_size:]
                        else:
                            self.audio_ring[:-n_samples] = self.audio_ring[n_samples:]
                            self.audio_ring[-n_samples:] = mono

                    # Get full buffer copy
                    with self.audio_lock:
                        audio_frame = self.audio_ring.copy()

                    # Calculate RMS level in dB
                    rms = np.sqrt(np.mean(audio_frame ** 2) + 1e-12)
                    db_level = float(20 * np.log10(rms + 1e-6))

                    # Process frame if not silent (> -58 dBFS)
                    if db_level < -58.0 or np.max(np.abs(audio_frame)) < 1e-4:
                        raw_chroma = np.zeros(12, dtype=np.float32)
                        self.smoothed_chroma = np.zeros(12, dtype=np.float32)
                        concert_chord = "N.C."
                        alto_chord = "N.C."
                        confidence = 0.0
                    else:
                        try:
                            # Pure NumPy / SciPy Chromagram Extraction (No Numba / Librosa DLL dependencies!)
                            raw_chroma = compute_chroma_scipy(audio_frame, sr=self.sample_rate)

                            # Exponential moving average smoothing
                            self.smoothed_chroma = (
                                self.smoothing_alpha * raw_chroma +
                                (1.0 - self.smoothing_alpha) * self.smoothed_chroma
                            )

                            # Detect chord
                            concert_chord, confidence, root_idx, quality = (
                                self.chord_detector.detect_chord(self.smoothed_chroma)
                            )

                            # Transpose for Alto Saxophone
                            if concert_chord != "N.C." and confidence > 0.40:
                                alto_chord = transpose_for_alto_sax(concert_chord, root_idx, quality)
                            else:
                                concert_chord = "N.C."
                                alto_chord = "N.C."
                                confidence = 0.0

                        except Exception as e:
                            print(f"Error in SciPy chromagram extraction: {e}")
                            concert_chord = "N.C."
                            alto_chord = "N.C."
                            confidence = 0.0
                            raw_chroma = np.zeros(12, dtype=np.float32)

                    # Fire UI callback
                    if self.on_chord_detected:
                        self.on_chord_detected(
                            concert_chord,
                            alto_chord,
                            confidence,
                            self.smoothed_chroma.copy(),
                            db_level
                        )

        except Exception as e:
            print(f"Audio engine loop exception: {e}")
            self.is_running = False
            if self.on_chord_detected:
                self.on_chord_detected("N.C.", "N.C.", 0.0, np.zeros(12), -100.0)
