import sys
import types
import unittest


def _install_stubs():
    # numpy stub with only APIs used in tested methods
    np = types.ModuleType("numpy")

    def interp(x, xp, fp):
        if x <= xp[0]:
            return fp[0]
        if x >= xp[-1]:
            return fp[-1]
        for i in range(len(xp) - 1):
            if xp[i] <= x <= xp[i + 1]:
                r = (x - xp[i]) / (xp[i + 1] - xp[i])
                return fp[i] + r * (fp[i + 1] - fp[i])
        return fp[-1]

    np.interp = interp
    np.asarray = lambda a, dtype=float: list(a)
    np.arange = lambda n, dtype=float: [float(i) for i in range(n)]
    np.clip = lambda x, lo, hi: max(lo, min(hi, x))
    np.diff = lambda arr: [arr[i + 1] - arr[i] for i in range(len(arr) - 1)]
    np.maximum = lambda arr, m: [max(v, m) for v in arr]
    np.mean = lambda arr: (sum(arr) / len(arr)) if arr else 0.0
    np.median = lambda arr: sorted(arr)[len(arr) // 2] if arr else 0.0
    np.ndarray = list
    np.sum = sum
    sys.modules["numpy"] = np

    # minimal music21 stub to satisfy module import
    music21 = types.ModuleType("music21")
    for sub in ["clef", "layout", "meter", "note", "stream", "tempo"]:
        setattr(music21, sub, types.SimpleNamespace())
    sys.modules["music21"] = music21


_install_stubs()

from audio_to_score_converter import AudioToScoreConverter, BeatAnalysis, NoteEvent


class AudioToScoreConverterTest(unittest.TestCase):
    def setUp(self):
        self.converter = AudioToScoreConverter(subdivision=4, beats_per_measure=4)
        self.beat_analysis = BeatAnalysis(
            beat_times=[0.0, 0.5, 1.02, 1.48, 2.01],
            downbeat_times=[0.0],
            beat_positions=[0.0, 1.0, 2.0, 3.0, 4.0],
            beats_per_measure=4,
        )

    def test_seconds_to_beat_interpolates_inside_range(self):
        beat = self.converter.seconds_to_beat(0.76, self.beat_analysis)
        self.assertAlmostEqual(beat, 1.5, places=2)

    def test_quantize_notes_maps_to_dynamic_grid(self):
        notes = [
            NoteEvent(midi=60, start_sec=0.03, end_sec=0.52, velocity=90),
            NoteEvent(midi=62, start_sec=1.01, end_sec=1.50, velocity=80),
        ]
        q = self.converter.quantize_notes(notes, self.beat_analysis)

        self.assertEqual(len(q), 2)
        self.assertAlmostEqual(q[0].start_beat, 0.0)
        self.assertAlmostEqual(q[0].duration_beats, 1.0)
        self.assertAlmostEqual(q[1].start_beat, 2.0)
        self.assertGreaterEqual(q[1].duration_beats, 0.75)


    def test_validate_audio_path_accepts_mp3(self):
        self.converter._validate_audio_path("input.mp3")

    def test_validate_audio_path_rejects_unsupported_suffix(self):
        with self.assertRaises(ValueError):
            self.converter._validate_audio_path("input.txt")


if __name__ == "__main__":
    unittest.main()
