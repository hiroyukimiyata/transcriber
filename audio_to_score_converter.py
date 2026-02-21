#!/usr/bin/env python3
"""Convert expressive/live audio performance into MusicXML using dynamic beat mapping.

Install dependencies:
    pip install numpy music21 basic-pitch madmom librosa soundfile

Usage:
    python audio_to_score_converter.py \
      --full-mix full_mix.mp3 \
      --isolated-part isolated_part.mp3 \
      --output output.musicxml
"""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from music21 import clef, layout, meter, note, stream, tempo


@dataclass
class BeatAnalysis:
    beat_times: np.ndarray
    downbeat_times: np.ndarray
    beat_positions: np.ndarray
    beats_per_measure: int


@dataclass
class NoteEvent:
    midi: int
    start_sec: float
    end_sec: float
    velocity: int


@dataclass
class QuantizedNote:
    midi: int
    start_beat: float
    duration_beats: float
    velocity: int


class AudioToScoreConverter:
    SUPPORTED_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}

    def __init__(self, subdivision: int = 4, beats_per_measure: int = 4, piano_split_midi: int = 60):
        self.subdivision = subdivision
        self.default_beats_per_measure = beats_per_measure
        self.piano_split_midi = piano_split_midi


    @classmethod
    def _validate_audio_path(cls, audio_path: str) -> None:
        suffix = Path(audio_path).suffix.lower()
        if suffix and suffix not in cls.SUPPORTED_AUDIO_SUFFIXES:
            supported = ", ".join(sorted(cls.SUPPORTED_AUDIO_SUFFIXES))
            raise ValueError(f"Unsupported audio format: {suffix}. Supported: {supported}")

    def analyze_beats(self, full_mix_path: str, prefer_madmom: bool = True) -> BeatAnalysis:
        """Analyze beat/downbeat timestamps from full mix, preserving tempo fluctuations."""
        if prefer_madmom and importlib.util.find_spec("madmom"):
            beat_times, downbeat_times, beats_per_measure = self._analyze_beats_madmom(full_mix_path)
        else:
            beat_times, downbeat_times, beats_per_measure = self._analyze_beats_librosa(full_mix_path)

        if len(beat_times) < 2:
            raise ValueError("Beat tracking failed: fewer than 2 beats detected.")

        beat_positions = np.arange(len(beat_times), dtype=float)
        return BeatAnalysis(
            beat_times=np.asarray(beat_times, dtype=float),
            downbeat_times=np.asarray(downbeat_times, dtype=float),
            beat_positions=beat_positions,
            beats_per_measure=beats_per_measure,
        )

    def _analyze_beats_madmom(self, full_mix_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
        from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
        from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor

        beat_activation = RNNBeatProcessor()(full_mix_path)
        beat_times = DBNBeatTrackingProcessor(fps=100)(beat_activation)

        downbeat_activation = RNNDownBeatProcessor()(full_mix_path)
        downbeat_sequence = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)(downbeat_activation)

        downbeat_times = downbeat_sequence[downbeat_sequence[:, 1] == 1, 0]
        if len(downbeat_times) >= 2:
            beat_counts = []
            for i in range(len(downbeat_times) - 1):
                n_beats = np.sum((beat_times >= downbeat_times[i]) & (beat_times < downbeat_times[i + 1]))
                if n_beats > 0:
                    beat_counts.append(n_beats)
            beats_per_measure = int(round(np.median(beat_counts))) if beat_counts else self.default_beats_per_measure
        else:
            beats_per_measure = self.default_beats_per_measure

        return beat_times, downbeat_times, beats_per_measure

    def _analyze_beats_librosa(self, full_mix_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
        import librosa

        y, sr = librosa.load(full_mix_path, sr=None, mono=True)
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        beats_per_measure = self.default_beats_per_measure
        downbeat_times = beat_times[::beats_per_measure] if len(beat_times) else np.array([])
        return beat_times, downbeat_times, beats_per_measure

    def extract_pitch_notes(self, isolated_part_path: str, min_confidence: float = 0.25) -> List[NoteEvent]:
        """Extract note events from isolated source with basic-pitch."""
        from basic_pitch.inference import predict

        _, midi_data, note_events = predict(isolated_part_path)

        events: List[NoteEvent] = []
        if note_events:
            for event in note_events:
                start_sec, end_sec, midi_pitch, confidence, *rest = event
                if float(confidence) < min_confidence:
                    continue
                velocity = int(np.clip((rest[0] if rest else confidence) * 127, 1, 127))
                events.append(
                    NoteEvent(
                        midi=int(midi_pitch),
                        start_sec=float(start_sec),
                        end_sec=float(end_sec),
                        velocity=velocity,
                    )
                )
        else:
            for instrument in midi_data.instruments:
                for n in instrument.notes:
                    events.append(
                        NoteEvent(
                            midi=int(n.pitch),
                            start_sec=float(n.start),
                            end_sec=float(n.end),
                            velocity=int(n.velocity),
                        )
                    )

        return sorted(events, key=lambda n: (n.start_sec, n.midi))

    def seconds_to_beat(self, t: float, beat_analysis: BeatAnalysis) -> float:
        bt = beat_analysis.beat_times
        bp = beat_analysis.beat_positions
        if t <= bt[0]:
            local_beat_sec = bt[1] - bt[0]
            return bp[0] + (t - bt[0]) / max(local_beat_sec, 1e-6)
        if t >= bt[-1]:
            local_beat_sec = bt[-1] - bt[-2]
            return bp[-1] + (t - bt[-1]) / max(local_beat_sec, 1e-6)
        return float(np.interp(t, bt, bp))

    @staticmethod
    def _snap_to_grid(value: float, step: float) -> float:
        return round(value / step) * step

    def quantize_notes(self, notes: Sequence[NoteEvent], beat_analysis: BeatAnalysis) -> List[QuantizedNote]:
        """Map absolute-time notes to dynamic beat grid and quantize to subdivision."""
        step = 1.0 / self.subdivision
        quantized: List[QuantizedNote] = []

        for n in notes:
            start_beat_raw = self.seconds_to_beat(n.start_sec, beat_analysis)
            end_beat_raw = self.seconds_to_beat(n.end_sec, beat_analysis)
            if end_beat_raw <= start_beat_raw:
                end_beat_raw = start_beat_raw + step

            start_beat = self._snap_to_grid(start_beat_raw, step)
            end_beat = self._snap_to_grid(end_beat_raw, step)
            if end_beat <= start_beat:
                end_beat = start_beat + step

            duration_beats = max(step, end_beat - start_beat)
            quantized.append(
                QuantizedNote(
                    midi=n.midi,
                    start_beat=float(start_beat),
                    duration_beats=float(duration_beats),
                    velocity=n.velocity,
                )
            )

        return sorted(quantized, key=lambda q: (q.start_beat, q.midi))

    def _measure_tempi(self, beat_analysis: BeatAnalysis) -> Dict[int, float]:
        beat_times = beat_analysis.beat_times
        bpm_per_beat = 60.0 / np.maximum(np.diff(beat_times), 1e-6)
        measure_tempi: Dict[int, float] = {}
        beats_per_measure = beat_analysis.beats_per_measure

        for i, bpm in enumerate(bpm_per_beat):
            measure_idx = (i // beats_per_measure) + 1
            measure_tempi.setdefault(measure_idx, [])
            measure_tempi[measure_idx].append(float(bpm))

        return {m: float(np.mean(v)) for m, v in measure_tempi.items()}

    def _build_single_staff_part(self, quantized_notes: Sequence[QuantizedNote], beat_analysis: BeatAnalysis) -> stream.Part:
        part = stream.Part(id="Part")
        part.append(meter.TimeSignature(f"{beat_analysis.beats_per_measure}/4"))

        for q in quantized_notes:
            n = note.Note(q.midi)
            n.quarterLength = q.duration_beats
            n.volume.velocity = q.velocity
            part.insert(q.start_beat, n)

        part.makeMeasures(inPlace=True)
        for m_idx, bpm in self._measure_tempi(beat_analysis).items():
            m = part.measure(m_idx)
            if m is not None:
                m.insert(0, tempo.MetronomeMark(number=round(bpm, 2)))
        return part

    def _build_piano_staff_parts(
        self, quantized_notes: Sequence[QuantizedNote], beat_analysis: BeatAnalysis
    ) -> Tuple[stream.PartStaff, stream.PartStaff]:
        right = stream.PartStaff(id="RH")
        left = stream.PartStaff(id="LH")
        ts = meter.TimeSignature(f"{beat_analysis.beats_per_measure}/4")
        right.append(ts)
        left.append(ts)
        right.append(clef.TrebleClef())
        left.append(clef.BassClef())

        for q in quantized_notes:
            n = note.Note(q.midi)
            n.quarterLength = q.duration_beats
            n.volume.velocity = q.velocity
            target = right if q.midi >= self.piano_split_midi else left
            target.insert(q.start_beat, n)

        right.makeMeasures(inPlace=True)
        left.makeMeasures(inPlace=True)
        measure_tempi = self._measure_tempi(beat_analysis)
        for m_idx, bpm in measure_tempi.items():
            rm = right.measure(m_idx)
            if rm is not None:
                rm.insert(0, tempo.MetronomeMark(number=round(bpm, 2)))
        return right, left

    def build_musicxml(
        self,
        quantized_notes: Sequence[QuantizedNote],
        beat_analysis: BeatAnalysis,
        output_musicxml_path: str,
        as_piano_grand_staff: bool = False,
    ) -> str:
        score = stream.Score(id="AudioToScore")

        if as_piano_grand_staff:
            right, left = self._build_piano_staff_parts(quantized_notes, beat_analysis)
            score.insert(0, right)
            score.insert(0, left)
            score.insert(0, layout.StaffGroup([right, left], symbol="brace", barTogether=True))
        else:
            score.insert(0, self._build_single_staff_part(quantized_notes, beat_analysis))

        output_path = str(Path(output_musicxml_path).with_suffix(".musicxml"))
        score.write("musicxml", fp=output_path)
        return output_path

    def convert(
        self,
        full_mix_path: str,
        isolated_part_path: str,
        output_musicxml_path: str,
        prefer_madmom: bool = True,
        as_piano_grand_staff: bool = False,
    ) -> str:
        self._validate_audio_path(full_mix_path)
        self._validate_audio_path(isolated_part_path)
        beat_analysis = self.analyze_beats(full_mix_path, prefer_madmom=prefer_madmom)
        note_events = self.extract_pitch_notes(isolated_part_path)
        quantized_notes = self.quantize_notes(note_events, beat_analysis)
        return self.build_musicxml(
            quantized_notes,
            beat_analysis,
            output_musicxml_path=output_musicxml_path,
            as_piano_grand_staff=as_piano_grand_staff,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert isolated instrument audio into MusicXML with dynamic beat mapping")
    parser.add_argument("--full-mix", required=True, help="Path to full mix audio (wav/mp3/...) for beat/downbeat analysis")
    parser.add_argument("--isolated-part", required=True, help="Path to isolated source audio (wav/mp3/...) for pitch extraction")
    parser.add_argument("--output", required=True, help="Output MusicXML file path")
    parser.add_argument("--subdivision", type=int, default=4, help="Quantization subdivision per beat (4=16th notes)")
    parser.add_argument("--beats-per-measure", type=int, default=4, help="Fallback beats per measure")
    parser.add_argument("--piano", action="store_true", help="Split into RH/LH staves by pitch")
    parser.add_argument("--no-madmom", action="store_true", help="Force librosa beat tracking")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    converter = AudioToScoreConverter(subdivision=args.subdivision, beats_per_measure=args.beats_per_measure)
    output = converter.convert(
        full_mix_path=args.full_mix,
        isolated_part_path=args.isolated_part,
        output_musicxml_path=args.output,
        prefer_madmom=not args.no_madmom,
        as_piano_grand_staff=args.piano,
    )
    print(f"MusicXML written: {output}")


if __name__ == "__main__":
    main()
