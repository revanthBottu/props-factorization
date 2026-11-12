"""Integration checks for the two-joint regular ProPS golf task."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from golf_rl_env import (ACTION_NAMES, DEFAULT_MATRIX, FEATURE_NAMES, GolfRLEnv,
                         MatrixGolfPolicy, positive_shot_reward)
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
    def test_gymnasium_api_and_universal_reset(self):
        env = GolfRLEnv(duration_s=1.2)
        try:
            check_env(env, skip_render_check=True)
            state, info = env.reset(seed=7)
            self.assertEqual(state.shape, (len(FEATURE_NAMES),))
            self.assertEqual(env.action_space.shape, (len(ACTION_NAMES),))
            self.assertAlmostEqual(info["position_angle_rad"], 0.0)
            _, _, terminated, truncated, info = env.step(np.zeros(2))
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIn("reward_components", info)
        finally:
            env.close()

        wrapped = gym.make("golf_rl_env:GolfSwing-v0", duration_s=1.2)
        try:
            wrapped.reset(seed=7)
        finally:
            wrapped.close()

    def test_reward_prefers_hits_and_face_center(self):
        miss = positive_shot_reward(0.1, False, 0.0, False)
        edge = positive_shot_reward(0.1, True, 0.1, False)
        center = positive_shot_reward(0.1, True, 1.0, False)
        sunk = positive_shot_reward(0.0, True, 1.0, True)
        self.assertEqual(sum(miss.values()), 0.0)
        self.assertGreater(sum(edge.values()), sum(miss.values()))
        self.assertGreater(sum(center.values()), sum(edge.values()))
        self.assertGreater(sum(sunk.values()), sum(center.values()))

    def test_two_joint_baseline_is_deterministic(self):
        env = GolfRLEnv()
        try:
            baseline = rollout(DEFAULT_MATRIX, env)
            repeated = rollout(DEFAULT_MATRIX, env)
            no_position = DEFAULT_MATRIX.copy()
            no_position[:, 0] = 0.0
            one_joint = rollout(no_position, env)
        finally:
            env.close()
        self.assertAlmostEqual(baseline["reward"], repeated["reward"])
        self.assertEqual(baseline["ball_hit"], repeated["ball_hit"])
        self.assertGreater(baseline["reward"], one_joint["reward"])
        self.assertTrue(baseline["ball_hit"])
        self.assertFalse(one_joint["ball_hit"])
        self.assertAlmostEqual(
            baseline["reward"], baseline["reward_components"]["total"]
        )

    def test_llm_receives_two_action_history_without_factorization(self):
        completions = FakeChatCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with tempfile.TemporaryDirectory() as directory:
            history = train(
                1, np.zeros_like(DEFAULT_MATRIX), model="test-model",
                output_dir=Path(directory), client=client,
                record_best_video=False, create_plots=False,
            )
            saved = json.loads((Path(directory) / "best_matrix.json").read_text())
        self.assertEqual(len(history), 2)
        self.assertEqual(saved["matrix"], DEFAULT_MATRIX.tolist())
        system_prompt = completions.request["messages"][0]["content"]
        user_prompt = completions.request["messages"][1]["content"]
        self.assertIn("regular ProPS", system_prompt)
        self.assertIn("do not factorize", system_prompt)
        self.assertIn("action_columns_in_order", user_prompt)
        self.assertIn("reward_components", user_prompt)
        np.testing.assert_array_equal(
            parse_matrix_json(
                "```json\n" + json.dumps({"matrix": DEFAULT_MATRIX.tolist()})
                + "\n```"
            ), DEFAULT_MATRIX
        )
        with self.assertRaises(ValueError):
            parse_matrix_json('{"matrix": [[1], [2]]}')
        with self.assertRaises(ValueError):
            MatrixGolfPolicy(np.full(DEFAULT_MATRIX.shape, np.nan))

    def test_asu_client_uses_env_configuration(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-only-key",
            "OPENAI_BASE_URL": "https://openai.rc.asu.edu/v1",
        }):
            client = make_llm_client()
            try:
                self.assertEqual(str(client.base_url),
                                 "https://openai.rc.asu.edu/v1/")
                self.assertEqual(client.api_key, "test-only-key")
            finally:
                client.close()


if __name__ == "__main__":
    unittest.main()
