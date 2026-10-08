import numpy as np
from chord_detector import ChordDetector, transpose_for_alto_sax, get_chord_note_info
from chroma import compute_chroma_scipy

_detector = ChordDetector(energy_threshold=0.02)
_smoothed = np.zeros(12, dtype=np.float32)


def reset_smoothing():
    global _smoothed
    _smoothed = np.zeros(12, dtype=np.float32)


def analyze_audio_chunk(samples, sr=22050, alpha=0.35):
    """
    Called from JS with a 1-D float array (mono, ~1.5 s ring buffer).
    Returns a JSON-serializable dict.
    """
    global _smoothed
    y = np.asarray(samples, dtype=np.float32)

    rms = float(np.sqrt(np.mean(y ** 2) + 1e-12))
    db_level = float(20 * np.log10(rms + 1e-6))

    if db_level < -58.0 or np.max(np.abs(y)) < 1e-4:
        _smoothed = np.zeros(12, dtype=np.float32)
        return {
            "concert_chord": "N.C.",
            "alto_chord": "N.C.",
            "confidence": 0.0,
            "chroma": _smoothed.tolist(),
            "db_level": db_level,
            "chord_notes": [],
            "active_pitch_indices": [],
        }

    raw = compute_chroma_scipy(y, sr=sr)
    _smoothed = alpha * raw + (1.0 - alpha) * _smoothed

    concert, conf, root, quality = _detector.detect_chord(_smoothed)
    if concert != "N.C." and conf > 0.40:
        alto = transpose_for_alto_sax(concert, root, quality)
        note_info = get_chord_note_info(concert)
        chord_notes = note_info.get("chord_notes", [])
        active_indices = list(note_info.get("active_pitch_indices", set()))
    else:
        concert, alto, conf = "N.C.", "N.C.", 0.0
        chord_notes = []
        active_indices = []

    return {
        "concert_chord": concert,
        "alto_chord": alto,
        "confidence": float(conf),
        "chroma": _smoothed.tolist(),
        "db_level": db_level,
        "chord_notes": chord_notes,
        "active_pitch_indices": active_indices,
    }


def analyze_file(samples, sr=22050, hop_seconds=0.15, buffer_seconds=1.5):
    """Offline chord timeline for uploaded files."""
    reset_smoothing()
    y = np.asarray(samples, dtype=np.float32)
    hop = int(sr * hop_seconds)
    buf = int(sr * buffer_seconds)
    timeline = []
    for start in range(0, max(1, len(y) - buf + 1), hop):
        chunk = y[start:start + buf]
        if len(chunk) < buf:
            chunk = np.pad(chunk, (0, buf - len(chunk)))
        result = analyze_audio_chunk(chunk, sr=sr)
        if result["concert_chord"] != "N.C.":
            result["time_sec"] = round(start / sr, 2)
            timeline.append(result)
    return timeline
