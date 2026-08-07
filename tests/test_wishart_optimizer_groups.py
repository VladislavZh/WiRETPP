from pathlib import Path
import sys
import unittest

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_wishart_overcomplete_backward_elimination import (
    _backbone_learning_rate_at_step,
    _make_optimizers,
)


class _Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(2, 2)
        self.head = torch.nn.Linear(2, 2)
        self.mixture_logits = torch.nn.Parameter(torch.zeros(2))


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_components = 2
        self.backbone = _Backbone()
        self.raw_mean_cholesky = torch.nn.Parameter(torch.eye(2))
        self.raw_alpha = torch.nn.Parameter(torch.zeros(()))

    @property
    def device(self):
        return self.raw_mean_cholesky.device

    @property
    def dtype(self):
        return self.raw_mean_cholesky.dtype

    def interaction_parameters(self):
        return [self.raw_alpha]


def _learning_rates(backbone_lr):
    optimizer, *_ = _make_optimizers(
        _Model(),
        neural_lr=1e-3,
        backbone_lr=backbone_lr,
        omega_lr=1e-2,
        alpha_lr=1e-4,
        weight_decay=1e-5,
        dual_lr=0.05,
    )
    return {
        group["group_name"]: group["lr"]
        for group in optimizer.param_groups
    }


class WishartOptimizerGroupTests(unittest.TestCase):
    def test_separate_backbone_learning_rate_keeps_head_rate(self):
        rates = _learning_rates(1e-4)
        self.assertEqual(rates["head"], 1e-3)
        self.assertEqual(rates["backbone"], 1e-4)

    def test_legacy_unified_learning_rate_remains_available(self):
        rates = _learning_rates(None)
        self.assertEqual(rates["head"], 1e-3)
        self.assertEqual(rates["backbone"], 1e-3)

    def test_overcomplete_backbone_lr_restores_only_after_target(self):
        values = [
            _backbone_learning_rate_at_step(
                step,
                final_prune_step=700,
                overcomplete_learning_rate=1e-5,
                post_target_learning_rate=1e-3,
                ramp_steps=200,
            )
            for step in (699, 700, 701, 800, 900, 1000)
        ]
        self.assertEqual(values[:2], [1e-5, 1e-5])
        self.assertAlmostEqual(values[2], 1e-5 + (1e-3 - 1e-5) / 200)
        self.assertAlmostEqual(values[3], 0.000505)
        self.assertEqual(values[4:], [1e-3, 1e-3])

    def test_zero_ramp_restores_backbone_immediately(self):
        self.assertEqual(
            _backbone_learning_rate_at_step(
                701,
                final_prune_step=700,
                overcomplete_learning_rate=3e-5,
                post_target_learning_rate=1e-3,
                ramp_steps=0,
            ),
            1e-3,
        )


if __name__ == "__main__":
    unittest.main()
