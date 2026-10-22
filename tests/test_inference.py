import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from evaluation.inference import artifact_subdir, load_inference_tensor, pick_tokens


class InferenceTests(unittest.TestCase):
    def test_loaded_metadata_does_not_reload_model_weights(self):
        checkpoint = {"model": {"large": torch.ones(10)}, "inference": {"unigram": torch.tensor([0.2, 0.8])}}
        path = Path("artifacts/generator.pt").resolve()
        with patch("evaluation.inference.torch.load", side_effect=AssertionError("unnecessary reload")):
            actual = load_inference_tensor(
                "generator.pt:inference.unigram", path.parent,
                device="cpu", dtype=torch.float64, checkpoint=checkpoint, checkpoint_path=path,
            )
        torch.testing.assert_close(actual, checkpoint["inference"]["unigram"].double())

    def test_separate_tensor_file_and_type_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.pt"
            torch.save(torch.tensor([0.25, 0.75]), path)
            actual = load_inference_tensor("stats.pt", directory, device="cpu", dtype=torch.float64)
            self.assertEqual(actual.tolist(), [0.25, 0.75])
            with self.assertRaises(TypeError):
                load_inference_tensor("model.pt:model", directory, device="cpu", dtype=torch.float32,
                                      checkpoint={"model": {}}, checkpoint_path=Path(directory) / "model.pt")

    def test_final_token_frequency_adjacency_and_deficit_policies(self):
        logits = torch.tensor([[[4., 3., 0.]] * 3])
        self.assertEqual(pick_tokens(logits).tolist(), [[0, 0, 0]])
        self.assertEqual(pick_tokens(logits, frequency=5).tolist(), [[0, 1, 2]])
        self.assertEqual(pick_tokens(logits, adjacency=[5, 1, 1]).tolist(), [[0, 1, 0]])
        self.assertEqual(pick_tokens(logits, frequency=5, allowance=torch.tensor([1., 0., 0.])).tolist(), [[0, 0, 0]])
        logits = torch.tensor([[[2., 1., 0.]] * 3])
        self.assertEqual(pick_tokens(logits, frequency=1, deficit=2,
                                     allowance=torch.tensor([0., 1., 0.])).tolist(), [[1, 1, 0]])

    def test_nested_artifact_directories_match_public_release(self):
        self.assertEqual(Path(artifact_subdir("sudoku-easy")), Path("sudoku/easy"))
        self.assertEqual(Path(artifact_subdir("lm1b-small")), Path("lm1b/small"))
        self.assertEqual(artifact_subdir("xsum"), "xsum")


if __name__ == "__main__":
    unittest.main()
