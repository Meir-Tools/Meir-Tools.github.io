import unittest
import numpy as np
from key_estimator import estimate_keys, analyze_chord_progression, get_scale_notes

class TestKeyEstimator(unittest.TestCase):
    def test_scale_notes(self):
        concert_c_maj, alto_c_maj = get_scale_notes(0, is_minor=False)
        self.assertEqual(concert_c_maj, ['C', 'D', 'E', 'F', 'G', 'A', 'B'])
        self.assertEqual(alto_c_maj, ['A', 'B', 'C#', 'D', 'E', 'F#', 'G#'])

        concert_a_min, alto_a_min = get_scale_notes(9, is_minor=True)
        self.assertEqual(concert_a_min, ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
        self.assertEqual(alto_a_min, ['F#', 'G#', 'A', 'B', 'C#', 'D', 'E'])

    def test_c_major_progression(self):
        # Progression: C - Am - F - G
        history = [
            {'concert_chord': 'C'},
            {'concert_chord': 'Am'},
            {'concert_chord': 'F'},
            {'concert_chord': 'G'}
        ]
        res = analyze_chord_progression(history)
        self.assertEqual(res['total_chords'], 4)
        self.assertEqual(res['unique_chords_count'], 4)

        top_key = res['key_candidates'][0]
        # Should predict C Major or A Minor
        self.assertIn(top_key['concert_key'], ['C Major', 'A Minor'])
        self.assertEqual(top_key['alto_key'], 'A Major' if top_key['concert_key'] == 'C Major' else 'F# Minor')

    def test_g_major_chroma_accumulator(self):
        # G Major pitches: G(7), B(11), D(2), C(0), D(2), E(4), F#(6)
        chroma = np.zeros(12)
        chroma[7] = 1.0  # G
        chroma[11] = 0.8 # B
        chroma[2] = 0.9  # D
        chroma[4] = 0.7  # E
        chroma[6] = 0.6  # F#

        keys = estimate_keys(chroma_accumulator=chroma, top_n=3)
        self.assertGreater(len(keys), 0)
        self.assertEqual(keys[0]['concert_key'], 'G Major')
        self.assertEqual(keys[0]['alto_key'], 'E Major')

if __name__ == '__main__':
    unittest.main()
