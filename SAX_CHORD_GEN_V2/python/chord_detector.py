import numpy as np
from typing import Tuple, Dict, Optional

# Pitch class names (12 semitones)
NOTE_NAMES_SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTE_NAMES_FLAT  = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

# Standard mapping from note name / accidental to pitch index (0..11)
NOTE_TO_INDEX = {
    'C': 0, 'C#': 1, 'Db': 1,
    'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4,
    'F': 5, 'F#': 6, 'Gb': 6,
    'G': 7, 'G#': 8, 'Ab': 8,
    'A': 9, 'A#': 10, 'Bb': 10,
    'B': 11
}

# Preferred note spelling for Alto Saxophone (Eb instrument) key signatures
ALTO_PREFERRED_NAMES = {
    0: 'C',
    1: 'C#',
    2: 'D',
    3: 'Eb',
    4: 'E',
    5: 'F',
    6: 'F#',
    7: 'G',
    8: 'G#',
    9: 'A',
    10: 'Bb',
    11: 'B'
}

# Define chord templates relative to root note (0 = root)
# Vector of length 12 with relative weights for pitch classes
CHORD_TEMPLATES: Dict[str, list[int]] = {
    '':       [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # Major triad (0, 4, 7)
    'm':      [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # Minor triad (0, 3, 7)
    '7':      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],  # Dominant 7th (0, 4, 7, 10)
    'maj7':   [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],  # Major 7th (0, 4, 7, 11)
    'm7':     [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],  # Minor 7th (0, 3, 7, 10)
    'dim':    [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],  # Diminished triad (0, 3, 6)
    'aug':    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # Augmented triad (0, 4, 8)
    'sus2':   [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Sus2 (0, 2, 7)
    'sus4':   [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # Sus4 (0, 5, 7)
}


class ChordDetector:
    def __init__(self, energy_threshold: float = 0.05):
        """
        :param energy_threshold: Minimum chromagram energy to consider sound as musical chord.
        """
        self.energy_threshold = energy_threshold
        self.templates = self._generate_all_templates()

    def _generate_all_templates(self) -> list[Tuple[str, int, str, np.ndarray]]:
        """
        Generates 12 * len(CHORD_TEMPLATES) target vectors normalized to unit norm.
        Returns a list of (full_chord_name, root_index, quality_suffix, template_vector).
        """
        all_templates = []
        for root_idx in range(12):
            root_name = NOTE_NAMES_SHARP[root_idx]
            for quality, rel_vector in CHORD_TEMPLATES.items():
                # Shift template relative vector by root_idx
                shifted_vector = np.roll(rel_vector, root_idx).astype(float)
                # Normalize vector
                norm = np.linalg.norm(shifted_vector)
                if norm > 0:
                    shifted_vector = shifted_vector / norm
                
                full_name = f"{root_name}{quality}"
                all_templates.append((full_name, root_idx, quality, shifted_vector))
        return all_templates

    def detect_chord(self, chroma: np.ndarray) -> Tuple[str, float, Optional[int], str]:
        """
        Detects the best matching chord from a 12-element chroma vector.

        :param chroma: 1D numpy array of shape (12,) representing pitch intensity.
        :return: Tuple of (chord_name, confidence, root_index, quality)
        """
        if chroma is None or len(chroma) != 12:
            return ("N.C.", 0.0, None, "")

        energy = np.sum(chroma)
        if energy < self.energy_threshold:
            return ("N.C.", 0.0, None, "")

        # Normalize incoming chroma
        norm = np.linalg.norm(chroma)
        if norm == 0:
            return ("N.C.", 0.0, None, "")

        norm_chroma = chroma / norm

        best_score = -1.0
        best_chord = "N.C."
        best_root = None
        best_quality = ""

        for full_name, root_idx, quality, template_vec in self.templates:
            # Cosine similarity
            score = np.dot(norm_chroma, template_vec)
            if score > best_score:
                best_score = score
                best_chord = full_name
                best_root = root_idx
                best_quality = quality

        confidence = max(0.0, float(best_score))
        return (best_chord, confidence, best_root, best_quality)


def transpose_for_alto_sax(chord_name: str, root_idx: Optional[int] = None, quality: str = "") -> str:
    """
    Transposes a concert pitch chord to Alto Saxophone (Eb instrument).
    Transposition rule: Alto Sax Pitch = (Concert Pitch + 9 semitones) % 12.

    :param chord_name: Concert chord name (e.g. "Cmaj7", "Fm", "N.C.")
    :param root_idx: Concert root note index (0..11) if available
    :param quality: Chord quality suffix (e.g. "maj7", "m", "7")
    :return: Transposed chord name for Alto Saxophone
    """
    if chord_name == "N.C." or chord_name == "" or root_idx is None:
        if chord_name and chord_name != "N.C." and not root_idx:
            # Parse chord_name if root_idx was not passed
            root_idx, quality = parse_chord_name(chord_name)
            if root_idx is None:
                return chord_name
        else:
            return "N.C."

    # Alto Sax transposition: +9 semitones
    alto_root_idx = (root_idx + 9) % 12
    alto_root_name = ALTO_PREFERRED_NAMES[alto_root_idx]
    
    return f"{alto_root_name}{quality}"


def parse_chord_name(chord_name: str) -> Tuple[Optional[int], str]:
    """
    Helper to parse root index and quality from chord string.
    """
    if not chord_name or chord_name == "N.C.":
        return None, ""
    
    # Match longest root note first
    for note in sorted(NOTE_TO_INDEX.keys(), key=len, reverse=True):
        if chord_name.startswith(note):
            root_idx = NOTE_TO_INDEX[note]
            quality = chord_name[len(note):]
            return root_idx, quality
            
    return None, ""


INTERVAL_NAMES: Dict[int, str] = {
    0: "Root",
    1: "b2",
    2: "2nd",
    3: "m3",
    4: "M3",
    5: "4th",
    6: "dim5",
    7: "5th",
    8: "aug5",
    9: "6th",
    10: "m7",
    11: "M7",
}


def get_chord_note_info(chord_name: str) -> Dict:
    """
    Analyzes a chord name and returns detailed info about its constituent notes,
    intervals, and pitch class indices for both Concert Pitch and Alto Saxophone Eb.
    """
    root_idx, quality = parse_chord_name(chord_name)
    if root_idx is None or chord_name == "N.C." or quality not in CHORD_TEMPLATES:
        return {
            'chord_name': chord_name or "N.C.",
            'root_idx': None,
            'quality': quality or "",
            'alto_chord': "N.C.",
            'chord_notes': [],
            'active_pitch_indices': set()
        }

    rel_vec = CHORD_TEMPLATES[quality]
    notes_info = []
    active_indices = set()

    for rel_idx, weight in enumerate(rel_vec):
        if weight > 0:
            pitch_idx = (root_idx + rel_idx) % 12
            active_indices.add(pitch_idx)
            concert_note = NOTE_NAMES_SHARP[pitch_idx]
            alto_pitch_idx = (pitch_idx + 9) % 12
            alto_note = ALTO_PREFERRED_NAMES[alto_pitch_idx]
            interval_name = INTERVAL_NAMES.get(rel_idx, f"+{rel_idx}")

            notes_info.append({
                'pitch_idx': pitch_idx,
                'rel_idx': rel_idx,
                'concert_note': concert_note,
                'alto_note': alto_note,
                'interval': interval_name
            })

    alto_chord = transpose_for_alto_sax(chord_name, root_idx, quality)

    return {
        'chord_name': chord_name,
        'root_idx': root_idx,
        'quality': quality,
        'alto_chord': alto_chord,
        'chord_notes': notes_info,
        'active_pitch_indices': active_indices
    }

