import unittest
import numpy as np
from chord_detector import ChordDetector, transpose_for_alto_sax, parse_chord_name, get_chord_note_info

class TestChordDetector(unittest.TestCase):
    def setUp(self):
        self.detector = ChordDetector(energy_threshold=0.01)

    def test_c_major_chord(self):
        # C major = C(0), E(4), G(7)
        chroma = np.zeros(12)
        chroma[0] = 1.0  # C
        chroma[4] = 0.8  # E
        chroma[7] = 0.9  # G
        
        chord, conf, root, quality = self.detector.detect_chord(chroma)
        self.assertEqual(chord, "C")
        self.assertEqual(root, 0)
        self.assertEqual(quality, "")
        self.assertGreater(conf, 0.8)

    def test_a_minor_chord(self):
        # A minor = A(9), C(0), E(4)
        chroma = np.zeros(12)
        chroma[9] = 1.0  # A
        chroma[0] = 0.8  # C
        chroma[4] = 0.9  # E
        
        chord, conf, root, quality = self.detector.detect_chord(chroma)
        self.assertEqual(chord, "Am")
        self.assertEqual(root, 9)
        self.assertEqual(quality, "m")

    def test_f_sharp_7_chord(self):
        # F#7 = F#(6), A#(10), C#(1), E(4)
        chroma = np.zeros(12)
        chroma[6] = 1.0  # F#
        chroma[10] = 0.7 # A#
        chroma[1] = 0.8  # C#
        chroma[4] = 0.6  # E
        
        chord, conf, root, quality = self.detector.detect_chord(chroma)
        self.assertEqual(chord, "F#7")
        self.assertEqual(root, 6)
        self.assertEqual(quality, "7")

    def test_alto_sax_transposition(self):
        # Concert C -> Alto Sax A
        self.assertEqual(transpose_for_alto_sax("C", 0, ""), "A")
        # Concert Am -> Alto Sax F#m
        self.assertEqual(transpose_for_alto_sax("Am", 9, "m"), "F#m")
        # Concert F7 -> Alto Sax D7
        self.assertEqual(transpose_for_alto_sax("F7", 5, "7"), "D7")
        # Concert Bb -> Alto Sax G
        self.assertEqual(transpose_for_alto_sax("Bb", 10, ""), "G")
        # Concert Eb -> Alto Sax C
        self.assertEqual(transpose_for_alto_sax("Eb", 3, ""), "C")

    def test_parse_chord_name(self):
        root, quality = parse_chord_name("Cmaj7")
        self.assertEqual(root, 0)
        self.assertEqual(quality, "maj7")
        
        root, quality = parse_chord_name("F#m")
        self.assertEqual(root, 6)
        self.assertEqual(quality, "m")
        
        root, quality = parse_chord_name("Bb7")
        self.assertEqual(root, 10)
        self.assertEqual(quality, "7")

    def test_get_chord_note_info(self):
        info = get_chord_note_info("Cmaj7")
        self.assertEqual(info['chord_name'], "Cmaj7")
        self.assertEqual(info['root_idx'], 0)
        self.assertEqual(info['quality'], "maj7")
        self.assertEqual(info['alto_chord'], "Amaj7")
        self.assertEqual(info['active_pitch_indices'], {0, 4, 7, 11})

        # Check notes details
        notes = info['chord_notes']
        self.assertEqual(len(notes), 4)
        # C -> Root (Concert C, Alto A)
        self.assertEqual(notes[0]['concert_note'], 'C')
        self.assertEqual(notes[0]['alto_note'], 'A')
        self.assertEqual(notes[0]['interval'], 'Root')
        # E -> M3 (Concert E, Alto C#)
        self.assertEqual(notes[1]['concert_note'], 'E')
        self.assertEqual(notes[1]['alto_note'], 'C#')
        self.assertEqual(notes[1]['interval'], 'M3')

if __name__ == '__main__':
    unittest.main()

