import gc
import itertools
import unittest
import weakref
from unittest.mock import patch

import torch

from tests.helpers import tiny_generator


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def compare_paths(self, model, *, cfg=0.0, stochastic=False):
        device = next(model.parameters()).device
        prefix = torch.tensor([[1, 2, 16, 16, 16], [3, 4, 5, 6, 7]], device=device)
        initial = None if model.query_level0 or model.predict_level0 else torch.tensor([[2], [3]], device=device)
        logits = []
        original = model._sample_level_raw

        def record(values, *args):
            logits.append(values.detach().clone())
            return original(values, *args)

        outputs, captures, passes = [], [], []
        for cached in (False, True):
            logits.clear()
            calls = []
            hook = model.transformer.blocks[0].register_forward_hook(lambda *args: calls.append(1))
            try:
                with patch.object(model, "_sample_level_raw", side_effect=record):
                    torch.manual_seed(42)
                    outputs.append(model.generate(
                        initial, prefix, use_kv_cache=cached,
                        temperature=1.0 if stochastic else 0.0,
                        top_k=0, top_p=1.0, cfg_scale=cfg, cfg_null_eos=15,
                    ))
                captures.append(list(logits))
                passes.append(len(calls))
            finally:
                hook.remove()
        self.assertTrue(torch.equal(*outputs))
        for eager, cached in zip(*captures):
            torch.testing.assert_close(eager, cached, rtol=2e-5, atol=2e-6)
        levels = len(model.levels)
        # Supplied level-0 codes need a separate prefix prefill in the cached path.
        self.assertEqual(passes, [levels, levels] if initial is None else [levels - 1, levels])
        self.assertIsNone(model._cfg)

    def test_cached_parity_across_generation_modes(self):
        for mode, rope, grouped, parallel, cfg in itertools.product(
            ("query", "predict", "supplied"), (False, True), (False, True), (False, True), (0.0, 0.7),
        ):
            with self.subTest(mode=mode, rope=rope, grouped=grouped, parallel=parallel, cfg=cfg):
                self.compare_paths(tiny_generator(mode=mode, rope=rope, grouped=grouped, parallel=parallel), cfg=cfg)

    def test_stochastic_parity_uses_the_same_random_stream(self):
        for mode in ("query", "predict", "supplied"):
            with self.subTest(mode=mode):
                self.compare_paths(tiny_generator(mode=mode), stochastic=True)

    def test_repeated_calls_release_inputs_without_cyclic_gc(self):
        model = tiny_generator()
        gc.collect()
        enabled = gc.isenabled()
        gc.disable()
        try:
            for cached in (False, True):
                for _ in range(3):
                    prefix = torch.tensor([[1, 2, 16, 16, 16]])
                    ref = weakref.ref(prefix)
                    model.generate(None, prefix, temperature=0, use_kv_cache=cached)
                    del prefix
                    self.assertIsNone(ref(), "generate retained an input until cyclic GC")
        finally:
            if enabled:
                gc.enable()
            gc.collect()

    def test_generation_recovers_after_a_cached_level_error(self):
        model = tiny_generator()
        prefix = torch.tensor([[1, 2, 16, 16, 16]])
        expected = model.generate(None, prefix, temperature=0, use_kv_cache=True)
        with patch.object(model, "_kv_level", side_effect=RuntimeError("injected failure")):
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                model.generate(None, prefix, temperature=0, use_kv_cache=True, cfg_scale=0.5, cfg_null_eos=15)
        actual = model.generate(None, prefix, temperature=0, use_kv_cache=True)
        self.assertTrue(torch.equal(actual, expected))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_cached_parity(self):
        matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        try:
            for mode in ("query", "predict", "supplied"):
                with self.subTest(mode=mode):
                    model = tiny_generator(mode=mode).cuda()
                    model.vqvae_local[0].cuda()
                    self.compare_paths(model, cfg=0.7)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul, cudnn


if __name__ == "__main__":
    unittest.main()
