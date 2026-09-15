import numpy as np
from typing import List, Dict, Tuple, Optional
from chord_detector import NOTE_NAMES_SHARP, ALTO_PREFERRED_NAMES, CHORD_TEMPLATES, parse_chord_name, transpose_for_alto_sax

# Krumhansl-Schmuckler Key Profiles for Major and Minor keys
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Scale interval offsets relative to root
MAJOR_SCALE_OFFSETS = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE_OFFSETS = [0, 2, 3, 5, 7, 8, 10]


def get_scale_notes(root_idx: int, is_minor: bool) -> Tuple[List[str], List[str]]:
    """
    Returns (concert_scale_notes, alto_scale_notes) for a given key root index and mode.
    """
    offsets = MINOR_SCALE_OFFSETS if is_minor else MAJOR_SCALE_OFFSETS
    concert_notes = []
    alto_notes = []

    for off in offsets:
        c_pitch = (root_idx + off) % 12
        concert_notes.append(NOTE_NAMES_SHARP[c_pitch])
        
        a_pitch = (c_pitch + 9) % 12
        alto_notes.append(ALTO_PREFERRED_NAMES[a_pitch])

    return concert_notes, alto_notes


def estimate_keys(
    history_entries: Optional[List[Dict]] = None,
    chroma_accumulator: Optional[np.ndarray] = None,
    top_n: int = 5
) -> List[Dict]:
    """
    Estimates key probabilities from chord history entries or accumulated chromagram vector.

    :param history_entries: List of chord history dicts (containing 'concert_chord', 'chroma', etc.)
    :param chroma_accumulator: 1D numpy array of shape (12,) accumulated pitch energy
    :param top_n: Number of top candidate keys to return
    :return: List of candidate key dictionaries ranked by probability score
    """
    pitch_weights = np.zeros(12)

    # 1. Accumulate pitch weights from chromagram vector if provided
    if chroma_accumulator is not None and len(chroma_accumulator) == 12:
        max_val = np.max(chroma_accumulator)
        if max_val > 0:
            pitch_weights += (chroma_accumulator / max_val)

    # 2. Accumulate pitch weights from history chord occurrences
    if history_entries:
        for entry in history_entries:
            chord_name = entry.get('concert_chord', '')
            if not chord_name or chord_name == "N.C.":
                continue
            
            root_idx, quality = parse_chord_name(chord_name)
            if root_idx is not None and quality in CHORD_TEMPLATES:
                template = np.roll(CHORD_TEMPLATES[quality], root_idx)
                pitch_weights += template * 1.5
            
            # Add entry's recorded chroma if available
            entry_chroma = entry.get('chroma')
            if entry_chroma is not None and len(entry_chroma) == 12:
                norm = np.linalg.norm(entry_chroma)
                if norm > 0:
                    pitch_weights += (entry_chroma / norm) * 0.8

    total_energy = np.sum(pitch_weights)
    if total_energy == 0:
        return []

    # Normalize pitch weights
    norm_weights = pitch_weights / np.linalg.norm(pitch_weights)

    candidates = []

    # Check 12 Major and 12 Minor keys
    for root_idx in range(12):
        root_name = NOTE_NAMES_SHARP[root_idx]
        alto_root_name = ALTO_PREFERRED_NAMES[(root_idx + 9) % 12]

        # Major Candidate
        shifted_maj = np.roll(MAJOR_PROFILE, root_idx)
        maj_norm = shifted_maj / np.linalg.norm(shifted_maj)
        score_maj = float(np.dot(norm_weights, maj_norm))

        concert_notes_maj, alto_notes_maj = get_scale_notes(root_idx, is_minor=False)

        candidates.append({
            'concert_key': f"{root_name} Major",
            'alto_key': f"{alto_root_name} Major",
            'raw_score': score_maj,
            'root_idx': root_idx,
            'is_minor': False,
            'concert_scale_notes': concert_notes_maj,
            'alto_scale_notes': alto_notes_maj
        })

        # Minor Candidate
        shifted_min = np.roll(MINOR_PROFILE, root_idx)
        min_norm = shifted_min / np.linalg.norm(shifted_min)
        score_min = float(np.dot(norm_weights, min_norm))

        concert_notes_min, alto_notes_min = get_scale_notes(root_idx, is_minor=True)

        candidates.append({
            'concert_key': f"{root_name} Minor",
            'alto_key': f"{alto_root_name} Minor",
            'raw_score': score_min,
            'root_idx': root_idx,
            'is_minor': True,
            'concert_scale_notes': concert_notes_min,
            'alto_scale_notes': alto_notes_min
        })

    # Sort candidates by score descending
    candidates.sort(key=lambda x: x['raw_score'], reverse=True)

    # Normalize top candidate scores to confidence percentages
    max_score = candidates[0]['raw_score'] if candidates else 1.0
    min_score = candidates[-1]['raw_score'] if candidates else 0.0
    score_range = max(1e-5, max_score - min_score)

    results = []
    for cand in candidates[:top_n]:
        norm_score = max(0.0, (cand['raw_score'] - min_score) / score_range)
        confidence_pct = int(min(99, max(15, norm_score * 100)))
        cand['confidence_pct'] = confidence_pct
        cand['confidence_str'] = f"{confidence_pct}%"
        results.append(cand)

    return results


def analyze_chord_progression(history_entries: List[Dict]) -> Dict:
    """
    Analyzes chord history progression to return chord frequency stats and estimated keys.
    """
    chord_counts: Dict[str, int] = {}
    total_valid_chords = 0

    for entry in history_entries:
        chord = entry.get('concert_chord', '')
        if chord and chord != "N.C.":
            chord_counts[chord] = chord_counts.get(chord, 0) + 1
            total_valid_chords += 1

    # Sort chords by count
    sorted_chords = sorted(chord_counts.items(), key=lambda x: x[1], reverse=True)
    chord_stats = []
    for chord, count in sorted_chords:
        pct = (count / total_valid_chords * 100) if total_valid_chords > 0 else 0
        alto_chord = transpose_for_alto_sax(chord)
        chord_stats.append({
            'concert_chord': chord,
            'alto_chord': alto_chord,
            'count': count,
            'percentage': pct,
            'percentage_str': f"{pct:.1f}%"
        })

    key_candidates = estimate_keys(history_entries=history_entries, top_n=5)

    return {
        'total_chords': total_valid_chords,
        'unique_chords_count': len(chord_counts),
        'chord_stats': chord_stats,
        'key_candidates': key_candidates
    }
