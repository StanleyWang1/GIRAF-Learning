from __future__ import annotations

import unittest

from giraf.learning.pipeline import evaluate
from giraf.learning.policy import Batch


class _BarePolicy:
    """Implements only the required Policy protocol: act, train_step, save."""

    def act(self, observation):
        raise NotImplementedError

    def train_step(self, batch):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError


class EvaluateTests(unittest.TestCase):
    def test_rejects_a_policy_without_evaluate(self) -> None:
        batch = Batch(observations={}, actions=[])
        with self.assertRaisesRegex(TypeError, "_BarePolicy"):
            evaluate(_BarePolicy(), [batch])


if __name__ == "__main__":
    unittest.main()
