"""Integration checks for the six-joint regular ProPS golf task."""

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

from golf_rl_env import (ACTION_NAMES, ANGLE_START, BIAS_START,
                         CLUB_TO_BALL_START, COLLISION_PENALTY_CAP,
                         DEFAULT_MATRIX, FEATURE_NAMES, GolfRLEnv,
                         MatrixGolfPolicy, N_TIME_PHASES, SPEED_START,
                         positive_shot_reward)
from run_golf_rl import make_llm_client, parse_matrix_json, rollout, train
from run_three_golf_tests import _first_available_series, _numbered_logdir


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
        env = GolfRLEnv(duration_s=2.0)
        try:
            check_env(env, skip_render_check=True)
            state, info = env.reset(seed=7)
            self.assertEqual(state.shape, (len(FEATURE_NAMES),))
            self.assertEqual(env.action_space.shape, (len(ACTION_NAMES),))
            self.assertEqual(env.sim.model.nu, 6)
            self.assertEqual(env.sim.model.neq, 0)
            self.assertEqual(env.sim.model.nkey, 1)
            self.assertGreaterEqual(env.sim.model.geom("club_mount").id, 0)
            self.assertGreaterEqual(env.sim.model.mesh("club_mount_mesh").id, 0)
            np.testing.assert_allclose(info["joint_angles_rad"], np.zeros(6))
            self.assertNotIn("joint_target_errors_rad", info)
            with self.assertRaises(KeyError):
                env.sim.model.key("address")

            club_rotation = np.asarray(
                env.sim.data.body("golf_club").xmat
            ).reshape(3, 3)
            wrist_rotation = np.asarray(
                env.sim.data.body("link6").xmat
            ).reshape(3, 3)
            np.testing.assert_allclose(
                club_rotation[:, 2], -wrist_rotation[:, 0], atol=1e-5
            )
            self.assertGreater(env.sim.model.geom("club_head").pos[0], 0.04)
            self.assertEqual(np.count_nonzero(state[:N_TIME_PHASES]), 1)
            np.testing.assert_allclose(
                state[CLUB_TO_BALL_START:], info["club_to_ball_vector_m"]
            )
            self.assertEqual(info["active_collision_pairs"], ())
            self.assertEqual(info["collision_pair_events"], {})
            _, _, terminated, truncated, info = env.step(np.zeros(6))
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIn("reward_components", info)
        finally:
            env.close()

        wrapped = gym.make("golf_rl_env:GolfSwing-v0", duration_s=2.0)
        try:
            wrapped.reset(seed=7)
        finally:
            wrapped.close()

    def test_reward_prefers_launch_quality_and_face_center(self):
        miss = positive_shot_reward(0.1, False, 0.0, False)
        touch = positive_shot_reward(0.1, True, 0.1, False)
        fast = positive_shot_reward(
            0.1, True, 0.1, False, launch_speed_m_s=2.0
        )
        directed = positive_shot_reward(
            0.1, True, 0.1, False, launch_speed_m_s=2.0,
            launch_direction_quality=1.0,
        )
        center = positive_shot_reward(
            0.1, True, 1.0, False, launch_speed_m_s=2.0,
            launch_direction_quality=1.0,
        )
        sunk = positive_shot_reward(
            0.0, True, 1.0, True, launch_speed_m_s=2.0,
            launch_direction_quality=1.0,
        )
        self.assertEqual(sum(miss.values()), 0.0)
        self.assertGreater(sum(touch.values()), sum(miss.values()))
        self.assertGreater(sum(fast.values()), sum(touch.values()))
        self.assertGreater(sum(directed.values()), sum(fast.values()))
        self.assertGreater(sum(center.values()), sum(directed.values()))
        self.assertGreater(sum(sunk.values()), sum(center.values()))
        self.assertEqual(sum(sunk.values()), 140.0)

    def test_baseline_is_stabilization_not_a_golf_solution(self):
        np.testing.assert_array_equal(DEFAULT_MATRIX[:N_TIME_PHASES], 0.0)
        np.testing.assert_array_equal(
            DEFAULT_MATRIX[ANGLE_START:ANGLE_START + 6], 0.0
        )
        np.testing.assert_array_equal(DEFAULT_MATRIX[CLUB_TO_BALL_START:], 0.0)
        np.testing.assert_array_equal(
            DEFAULT_MATRIX[BIAS_START:BIAS_START + 6], np.eye(6)
        )
        self.assertTrue(np.all(np.diag(
            DEFAULT_MATRIX[SPEED_START:SPEED_START + 6]
        ) < 0))
        env = GolfRLEnv()
        try:
            baseline = rollout(DEFAULT_MATRIX, env)
            repeated = rollout(DEFAULT_MATRIX, env)
        finally:
            env.close()
        self.assertAlmostEqual(baseline["reward"], repeated["reward"])
        self.assertEqual(baseline["ball_hit"], repeated["ball_hit"])
        self.assertFalse(baseline["ball_hit"])
        self.assertAlmostEqual(
            baseline["reward"], baseline["reward_components"]["total"]
        )

    def test_effort_collision_cap_and_settle_termination(self):
        self.assertGreater(COLLISION_PENALTY_CAP, 140.0)
        env = GolfRLEnv(duration_s=2.0)
        try:
            env.reset()
            _, _, _, _, info = env.step(env.action_space.high)
            self.assertLess(info["reward_components"]["effort_penalty"], 0.0)

            env.reset()
            env.sim.collision_events = 100
            env.sim.collision_time_s = 100.0
            _, _, _, _, info = env.step(np.zeros(6))
            self.assertEqual(
                info["reward_components"]["collision_penalty"],
                -COLLISION_PENALTY_CAP,
            )

            state, _ = env.reset()
            env.sim.hit = True
            policy = MatrixGolfPolicy(DEFAULT_MATRIX)
            terminated = truncated = False
            while not (terminated or truncated):
                state, _, terminated, truncated, info = env.step(
                    policy.action(state)
                )
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info["ball_settled"])
            self.assertEqual(info["reward_components"]["contact"], 10.0)
        finally:
            env.close()

    def test_llm_receives_six_action_history_without_factorization(self):
        completions = FakeChatCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with tempfile.TemporaryDirectory() as directory:
            history = train(
                1, DEFAULT_MATRIX, model="test-model",
                output_dir=Path(directory), client=client,
                record_best_video=False, create_plots=False,
            )
            saved = json.loads((Path(directory) / "best_matrix.json").read_text())
        self.assertEqual(len(history), 2)
        self.assertEqual(saved["matrix"], DEFAULT_MATRIX.tolist())
        system_prompt = completions.request["messages"][0]["content"]
        user_prompt = completions.request["messages"][1]["content"]
        self.assertIn("regular ProPS", system_prompt)
        self.assertIn("six-joint", system_prompt)
        self.assertIn("do not factorize", system_prompt)
        self.assertIn("No address pose", system_prompt)
        self.assertIn("discover positioning", system_prompt)
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

    def test_batch_log_names_follow_yaml_base_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "test"
            self.assertEqual(_numbered_logdir(base, 0), base)
            self.assertEqual(_numbered_logdir(base, 2).name, "test_2")
            start, paths = _first_available_series(base, 3)
            self.assertEqual(start, 0)
            self.assertEqual([path.name for path in paths],
                             ["test", "test_1", "test_2"])
            base.mkdir()
            start, paths = _first_available_series(base, 3)
            self.assertEqual(start, 1)
            self.assertEqual([path.name for path in paths],
                             ["test_1", "test_2", "test_3"])


if __name__ == "__main__":
    unittest.main()
