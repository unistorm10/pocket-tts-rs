"""Regression tests for independent speech-quality acceptance gates.

No checkpoints, reference recordings, GPU, or third-party ASR downloads needed.
"""
import importlib.util
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

MODULE = Path(__file__).with_name("audio_qa.py")
spec = importlib.util.spec_from_file_location("pocket_audio_qa", MODULE)
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)


class QualityGateTests(unittest.TestCase):
    def test_identical_transcription_has_zero_error(self):
        words = qa.words("Testing real Candle synthesis with a cloned voice.")
        self.assertEqual(qa.edit_distance(words, words), 0)

    def test_repetition_is_not_accepted_as_speech(self):
        reference = qa.words("Testing real Candle synthesis with a cloned voice.")
        repeated = qa.words("T-" * 100)
        wer = qa.edit_distance(reference, repeated) / len(reference)
        self.assertGreater(wer, 0.5)

    def test_empty_transcription_fails_intelligibility(self):
        reference = qa.words("Testing real Candle synthesis with a cloned voice.")
        self.assertEqual(qa.edit_distance(reference, []), len(reference))

    def test_nonsilent_pcm_does_not_prove_speech(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "noise.wav"
            import math
            import struct
            pcm = [int(5000 * math.sin(i * 0.2)) for i in range(24000)]
            with wave.open(str(path), "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(24000)
                out.writeframes(struct.pack("<" + "h" * len(pcm), *pcm))
            _, checks = qa.pcm_analysis(path)
            self.assertTrue(all(checks.values()))
            # Acoustic checks alone cannot establish words or speaker identity.
            self.assertEqual(len(checks), 4)


if __name__ == "__main__":
    unittest.main()
