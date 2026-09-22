import json
from pathlib import Path
import tempfile
import unittest

from echoprime_track.check_grpo_run import check_run
from echoprime_track.prompts import SYSTEM_PROMPT


class TestRunGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "rollouts").mkdir()

    def write_run(self, *, stale=False, grad=0.1, clip=0.0):
        metrics = []
        for step in (1, 2):
            metrics.append({"step": step, "data": {
                "actor/grad_norm": grad, "response_length/clip_ratio": clip,
                "timing_s/update_weights": 1.0}})
            rows = [{"input": ("stale" if stale else SYSTEM_PROMPT) + f"\nq{i}\n<think>\n",
                     "output": "reasoning</think>No.", "score": float(j)}
                    for i in range(4) for j in range(2)]
            (self.root / "rollouts" / f"{step}.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows))
        (self.root / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in metrics))

    def test_valid_run_passes_without_claiming_policy_tool_use(self):
        self.write_run()
        result = check_run(self.root)
        self.assertEqual(result["gate"], "PASS")
        self.assertFalse(result["policy_tool_use_observed"])

    def test_stale_actual_rollout_is_rejected(self):
        self.write_run(stale=True)
        with self.assertRaisesRegex(ValueError, "Actual rollout input"):
            check_run(self.root)

    def test_zero_gradients_are_rejected(self):
        self.write_run(grad=0)
        with self.assertRaisesRegex(ValueError, "learning signal"):
            check_run(self.root)

    def test_excessive_truncation_is_rejected(self):
        self.write_run(clip=0.5)
        with self.assertRaisesRegex(ValueError, "excessive truncation"):
            check_run(self.root)
