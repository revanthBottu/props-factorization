"""Exercise the golf task through the shared ProPS runner without an API call."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import yaml
from jinja2 import Template

from agent.llm_num_optim_linear_policy import LLMNumOptimAgent
from agent.policy.llm_brain_linear_policy import LLMBrain
from runner.llm_num_optim_runner import run_training_loop


HERE = Path(__file__).resolve().parent


class GolfPropsIntegrationTests(unittest.TestCase):
    def test_asu_provider_uses_openai_compatible_chat(self):
        completion = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="params[0]: 1.1")
        )])
        fake_client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: completion)
        ))
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-only",
            "OPENAI_BASE_URL": "https://openai.rc.asu.edu/v1",
        }), \
             patch("agent.policy.llm_brain_linear_policy.OpenAI", return_value=fake_client) as client:
            brain = LLMBrain(Template(""), Template(""), "asu/gemma4-31b-it")
            brain.add_llm_conversation("test prompt", "user")
            self.assertEqual(brain.query_llm(), "params[0]: 1.1")
        self.assertEqual(brain.model_group, "asu")
        self.assertEqual(client.call_args.kwargs["api_key"], "test-only")
        self.assertEqual(client.call_args.kwargs["base_url"],
                         "https://openai.rc.asu.edu/v1")

    def test_golf_config_uses_shared_runner_and_log_layout(self):
        config = yaml.safe_load(
            (HERE / "configs/golf/golf_props.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(config["task"], "cont_space_llm_num_optim")
        self.assertFalse(config["use_factorized_policy"])
        self.assertEqual(config["llm_si_template_name"], "num_optim.j2")
        with gym.make(config["gym_env_name"]) as env:
            self.assertEqual(env.observation_space.shape, (config["dim_states"],))
            self.assertEqual(env.action_space.shape, (config["dim_actions"],))
        matrix = np.asarray(config["initial_matrix"], dtype=float)
        self.assertEqual(matrix.shape, (26, 6))

        values = matrix.reshape(-1).copy()
        values[5] += 0.1  # ProPS rejects an exact repeat of the warmup matrix.
        reply = ", ".join(
            f"params[{index}]: {value}" for index, value in enumerate(values)
        ) + "\nExplore a small phase-zero joint5 torque."

        with tempfile.TemporaryDirectory() as folder:
            config["logdir"] = folder
            config["num_episodes"] = 1
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only"}), \
                 patch.object(LLMBrain, "query_llm", return_value=reply), \
                 patch.object(LLMNumOptimAgent, "record_best_episode"):
                run_training_loop(**config)

            root = Path(folder)
            self.assertTrue((root / "overall_log.txt").is_file())
            self.assertTrue((root / "warmup/warmup_rollout_0.txt").is_file())
            episode = root / "episode_0"
            for name in (
                "parameters.txt", "parameters_reasoning.txt",
                "training_rollout.txt", "reward_progress.png",
                "policy_heatmap_ep0.png",
            ):
                self.assertTrue((episode / name).is_file(), name)
            self.assertTrue((root / "policy_heatmaps.gif").is_file())
            self.assertIn("Iteration, CPU Time, API Time", (root / "overall_log.txt").read_text())
            prompt_log = (episode / "parameters_reasoning.txt").read_text()
            self.assertIn("MAX_STEPS (1)", prompt_log)
            self.assertIn("[-50.0, 50.0]", prompt_log)
            self.assertIn("six action columns", prompt_log)


if __name__ == "__main__":
    unittest.main()
