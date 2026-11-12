"""Integration checks for the golf policy search loop (no network needed)."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from golf_rl_env import DEFAULT_MATRIX, GolfRLEnv, MatrixGolfPolicy, distance_reward
from run_golf_rl import make_llm_client, parse_matrix_json, rollout, train


class FakeChatCompletions:
    def __init__(self):
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        content = json.dumps({"matrix": DEFAULT_MATRIX.tolist()})
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content))])


class GolfRLTests(unittest.TestCase):
    def test_reward_and_reset(self):
        env = GolfRLEnv()
        try:
            zero = rollout(np.zeros_like(DEFAULT_MATRIX), env)
            baseline = rollout(DEFAULT_MATRIX, env)
            repeated = rollout(DEFAULT_MATRIX, env)
        finally:
            env.close()
        self.assertGreater(baseline["reward"], zero["reward"])
        self.assertAlmostEqual(baseline["reward"],
                               distance_reward(baseline["distance_to_hole_m"]))
        self.assertEqual(baseline["reward"], repeated["reward"])
        self.assertLess(abs(baseline["wrist_speed_rad_s"]), 0.01)
        self.assertAlmostEqual(distance_reward(0), 50.0)

    def test_llm_revision_receives_history_and_can_improve_score(self):
        completions = FakeChatCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with tempfile.TemporaryDirectory() as directory:
            history = train(1, np.zeros_like(DEFAULT_MATRIX), model="test-model",
                            output_dir=Path(directory), client=client)
            saved = json.loads((Path(directory) / "best_matrix.json").read_text())
        self.assertGreater(history[1]["reward"], history[0]["reward"])
        self.assertEqual(saved["matrix"], DEFAULT_MATRIX.tolist())
        prompt = completions.request["messages"][1]["content"]
        self.assertIn("recent_trials", prompt)
        self.assertIn("feature_names_in_row_order", prompt)
        self.assertNotIn("response_format", completions.request)
        np.testing.assert_array_equal(
            parse_matrix_json("```json\n" + json.dumps({"matrix": DEFAULT_MATRIX.tolist()})
                              + "\n```"), DEFAULT_MATRIX)
        with self.assertRaises(ValueError):
            parse_matrix_json('{"matrix": [[1], [2]]}')
        with self.assertRaises(ValueError):
            MatrixGolfPolicy([[float("nan")]] * 7)

    def test_asu_client_uses_env_configuration(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only-key",
                                  "OPENAI_BASE_URL": "https://openai.rc.asu.edu/v1"}):
            client = make_llm_client()
            try:
                self.assertEqual(str(client.base_url),
                                 "https://openai.rc.asu.edu/v1/")
                self.assertEqual(client.api_key, "test-only-key")
            finally:
                client.close()


if __name__ == "__main__":
    unittest.main()
