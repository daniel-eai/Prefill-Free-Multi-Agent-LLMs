import unittest

import torch

from heterofold import alignment, calibration, recolor
from heterofold.evaluation import longbench, short_context
from heterofold.evaluation.runtime import shard_range


def offsets(ends):
    starts = [0] + list(ends[:-1])
    return list(zip(starts, ends))


class TokenAlignmentTest(unittest.TestCase):
    qwen = offsets([3, 16, 23, 24])
    ministral = offsets([3, 9, 12, 16, 23, 24])

    def test_coarse_sender_to_fine_receiver(self):
        self.assertEqual(
            alignment.receiver_to_sender_indices(self.qwen, self.ministral),
            [0, 0, 0, 1, 2, 3])

    def test_fine_sender_to_coarse_receiver(self):
        self.assertEqual(
            alignment.receiver_to_sender_indices(self.ministral, self.qwen),
            [0, 3, 4, 5])

    def test_exact_pairs_only_use_shared_boundaries(self):
        self.assertEqual(
            alignment.exact_end_pairs(self.qwen, self.ministral),
            [(0, 0), (1, 3), (2, 4), (3, 5)])

    def test_layer_stencil(self):
        taps = alignment.stencil_taps(32, 36, span=1, stride=4)
        self.assertEqual(taps[:4], [[0, 0, 4], [0, 1, 5], [0, 2, 6], [0, 3, 7]])
        self.assertEqual(taps[-1], [27, 31, 31])
        self.assertEqual([row[1] for row in taps],
                         alignment.proportional_layers(32, 36))


class RecolorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        count, source_width, target_width = 20000, 24, 8
        latent = torch.randn(count, target_width)
        self.x = (latent @ torch.randn(target_width, source_width)
                  + 0.3 * torch.randn(count, source_width) + 2.0)
        self.y = latent @ torch.randn(target_width, target_width) - 1.0
        self.stats = (self.x.sum(0), self.y.sum(0), self.x.T @ self.x,
                      self.x.T @ self.y, self.y.T @ self.y)
        self.count = float(count)

    def test_recolor_matches_receiver_moments(self):
        weight, bias, rms = recolor.fit_affine(
            *self.stats, self.count, device="cpu")
        mapped = (self.x / rms) @ weight + bias
        self.assertTrue(torch.allclose(
            mapped.mean(0), self.y.mean(0), atol=1e-3))
        relative = ((torch.cov(mapped.T) - torch.cov(self.y.T)).norm()
                    / torch.cov(self.y.T).norm())
        self.assertLess(float(relative), 0.02)

    def test_corpus_balancing_equalizes_token_mass(self):
        def corpus(tokens):
            role = {field: [torch.full((2,), float(tokens))]
                    for field in recolor.FIELDS}
            return {"roles": {"k": role, "v": role}, "N": tokens, "texts": 1,
                    "Lr": 1}
        combined = recolor.balance_corpora({"a": corpus(100), "b": corpus(900)})
        self.assertEqual(combined["N"], 1000)
        self.assertAlmostEqual(combined["moment_scales"]["a"], 5.0)
        self.assertAlmostEqual(combined["moment_scales"]["b"], 500 / 900)


class FoldingTest(unittest.TestCase):
    def test_fold_equals_runtime_correction(self):
        torch.manual_seed(0)
        width, rank, source = 16, 4, 48
        base = {"A_k": [torch.randn(source, width)], "b_k": [torch.randn(width)],
                "A_v": [torch.randn(source, width)], "b_v": [torch.randn(width)]}
        correction = calibration.LayerCorrection(width, rank)
        with torch.no_grad():
            for role in "kv":
                getattr(correction, f"right_{role}").normal_()
        folded = calibration.fold(base, [correction], {"rank": rank})
        x = torch.randn(5, source)
        for role, apply in (("k", correction.key), ("v", correction.value)):
            expected = apply(x @ base[f"A_{role}"][0] + base[f"b_{role}"][0])
            actual = x @ folded[f"A_{role}"][0] + folded[f"b_{role}"][0]
            self.assertTrue(torch.allclose(expected, actual, atol=1e-4))

    def test_correction_starts_at_identity(self):
        correction = calibration.LayerCorrection(16, 4)
        x = torch.randn(3, 16)
        self.assertTrue(torch.equal(correction.key(x), x))
        self.assertTrue(torch.equal(correction.value(x), x))


class ScoringTest(unittest.TestCase):
    def test_gsm8k_answer_extraction(self):
        self.assertEqual(short_context.extract_numeric_answer(
            "so the total is \\boxed{1,234}."), "1234")
        self.assertEqual(short_context.extract_numeric_answer(
            "#### 18.0"), "18")
        self.assertEqual(short_context.extract_numeric_answer(
            "She pays $7.50 in total"), "7.5")

    def test_longbench_f1(self):
        self.assertAlmostEqual(longbench.qa_f1("The Eiffel Tower", "eiffel tower"), 1.0)
        self.assertEqual(longbench.qa_f1("Paris", "London"), 0.0)

    def test_locomo_prediction_cleanup(self):
        self.assertEqual(longbench.locomo_clean("Answer: 7 May 2023\nextra"),
                         "7 May 2023")

    def test_shards_cover_every_item_once(self):
        ranges = [shard_range(1172, f"{i}/4", None, None) for i in range(4)]
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[-1][1], 1172)
        for (_, end), (start, _) in zip(ranges, ranges[1:]):
            self.assertEqual(end, start)


if __name__ == "__main__":
    unittest.main()
