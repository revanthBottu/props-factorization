from agent.policy.linear_policy_no_bias import LinearPolicy as LinearPolicyNoBias
from agent.policy.linear_policy import LinearPolicy
from agent.policy.replay_buffer import EpisodeRewardBufferNoBias
from agent.policy.llm_brain_linear_policy import LLMBrain
from world.base_world import BaseWorld
import random
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
import numpy as np
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os
import platform

# Enable headless rendering for video recording on servers without display
# Only use 'egl' on Linux; Windows uses 'glfw' by default
if platform.system() == 'Linux':
    os.environ['MUJOCO_GL'] = 'egl'

import gymnasium as gym
from gymnasium.wrappers import RecordVideo


class LLMNumOptimAgent:
    def __init__(
        self,
        logdir,
        dim_action,
        dim_state,
        max_traj_count,
        max_traj_length,
        llm_si_template,
        llm_output_conversion_template,
        llm_model_name,
        num_evaluation_episodes,
        bias,
        optimum,
        search_step_size,
        use_factorized_policy=False,
        factor_rank=None,
        decomposition_type: str = "lu",
        frozen_factor=None,
        enable_alternating_lu_schedule: bool = False,
        lu_schedule_l_episodes: int = 5,
        lu_schedule_u_iterations: int = 3,
        seed: int = None,
    ):
        self.start_time = time.process_time()
        self.api_call_time = 0
        self.total_steps = 0
        self.total_episodes = 0
        self.dim_action = dim_action
        self.dim_state = dim_state
        self.bias = bias
        self.optimum = optimum
        self.search_step_size = search_step_size
        self.use_factorized_policy = use_factorized_policy
        self.decomposition_type = str(decomposition_type or "lu").lower()
        self.allowed_factor_names = {
            "lu": ("L", "U"),
            "qr": ("Q", "R"),
            "svd": ("U", "S", "Vt"),
        }
        if self.decomposition_type not in self.allowed_factor_names:
            raise ValueError(
                "decomposition_type must be one of "
                f"{tuple(self.allowed_factor_names.keys())}, got: {decomposition_type!r}"
            )

        self.frozen_factor = self._canonicalize_factor_name(frozen_factor)
        if self.frozen_factor is not None and self.frozen_factor not in self.allowed_factor_names[self.decomposition_type]:
            raise ValueError(
                f"frozen_factor={frozen_factor!r} is invalid for decomposition_type={self.decomposition_type!r}. "
                f"Allowed: {self.allowed_factor_names[self.decomposition_type]} or None"
            )

        self.enable_alternating_lu_schedule = bool(enable_alternating_lu_schedule)
        self.lu_schedule_l_episodes = int(lu_schedule_l_episodes)
        self.lu_schedule_u_iterations = int(lu_schedule_u_iterations)
        if self.enable_alternating_lu_schedule:
            if not self.use_factorized_policy:
                raise ValueError("enable_alternating_lu_schedule requires use_factorized_policy=True")
            if self.decomposition_type != "lu":
                raise ValueError("enable_alternating_lu_schedule currently supports decomposition_type='lu' only")
            if self.lu_schedule_l_episodes <= 0 or self.lu_schedule_u_iterations <= 0:
                raise ValueError("lu_schedule_l_episodes and lu_schedule_u_iterations must both be positive integers")
        # Deterministic seed for episode rollouts. If None, default to 42.
        self.seed = seed if seed is not None else 42
        self.force_new_matrix_reward_threshold = -100.0
        # Forced exploration strictness: each editable index must move by at least
        # this absolute amount relative to recent attempts.
        self.force_exploration_min_index_delta = 0.35
        self.force_exploration_reference_count = 5
        self.force_exploration_max_llm_attempts = 3

        if not self.bias:
            param_count = dim_action * dim_state
        else:
            param_count = dim_action * dim_state + dim_action
        self.rank = param_count
        
        # Setup factor rank for two-matrix policy representation
        print(
            "[DEBUG] "
            f"use_factorized_policy={use_factorized_policy}, "
            f"factor_rank={factor_rank}, dim_state={dim_state}, dim_action={dim_action}, "
            f"decomposition_type={self.decomposition_type}, frozen_factor={self.frozen_factor}"
        )
        if use_factorized_policy:
            if factor_rank is None:
                self.factor_rank = max(1, min(dim_state, dim_action) // 2)
            else:
                self.factor_rank = factor_rank
        else:
            self.factor_rank = None
        print(f"[DEBUG] self.factor_rank set to {self.factor_rank}")

        if not self.bias:
            self.policy = LinearPolicyNoBias(
                dim_actions=dim_action, dim_states=dim_state
            )
        else:
            self.policy = LinearPolicy(
                dim_actions=dim_action, 
                dim_states=dim_state,
                use_factorized_policy=use_factorized_policy,
                factor_rank=self.factor_rank,
                decomposition_type=self.decomposition_type,
            )
        self.replay_buffer = EpisodeRewardBufferNoBias(max_size=max_traj_count)
        self.llm_brain = LLMBrain(
            llm_si_template, llm_output_conversion_template, llm_model_name
        )
        self.logdir = logdir
        self.num_evaluation_episodes = num_evaluation_episodes
        self.training_episodes = 0
        
        # Track rewards for visualization - separate warmup and training
        self.warmup_rewards = []
        self.training_rewards = []
        
        # Track best reward for conditional video recording
        self.best_reward = -float('inf')

        # Alternating LU schedule state:
        # keep the best-performing candidate within each phase and use it when
        # switching phases to decide which matrix should be frozen next.
        self._lu_schedule_prev_phase = None
        self._lu_phase_best_reward = float('-inf')
        self._lu_phase_best_components = None

        # Quality shaping metadata from the most recently parsed factor proposal.
        self._current_matrix_structure_penalty = 0.0
        self._current_matrix_quality_notes = []

        if self.bias:
            self.dim_state += 1

    def _is_groq_mode(self) -> bool:
        return getattr(self.llm_brain, "model_group", None) == "groq"

    def _canonicalize_factor_name(self, factor_name):
        if factor_name is None:
            return None

        if not self.use_factorized_policy:
            return None

        if hasattr(self, "policy") and hasattr(self.policy, "canonicalize_factor_name"):
            normalized = self.policy.canonicalize_factor_name(factor_name)
            if normalized is not None:
                return normalized

        stripped = str(factor_name).strip()
        if stripped in self.allowed_factor_names.get(self.decomposition_type, ()):
            return stripped

        lowered = stripped.lower()
        for candidate in self.allowed_factor_names.get(self.decomposition_type, ()): 
            if lowered == candidate.lower():
                return candidate
        return None

    def _factor_names(self):
        if not self.use_factorized_policy:
            return []
        return list(self.allowed_factor_names[self.decomposition_type])

    def _factor_shapes(self):
        if not self.use_factorized_policy:
            return {}
        return self.policy.get_factor_shapes()

    def _factor_is_matrix(self, factor_name):
        shape = self._factor_shapes().get(factor_name)
        return isinstance(shape, tuple) and len(shape) == 2

    def _select_prompt_replay_entries(self, replay_buffer: EpisodeRewardBufferNoBias):
        """Select replay entries for prompting.

        For Groq, keep only the latest 4 entries and the top 3 by reward.
        For other providers, keep the full replay buffer.
        """
        entries = list(replay_buffer.buffer)
        if not self._is_groq_mode() or len(entries) <= 7:
            return entries

        recent_keep = 4
        best_keep = 3
        total = len(entries)

        recent_indices = set(range(max(0, total - recent_keep), total))

        scored_indices = []
        for idx, entry in enumerate(entries):
            reward = float("-inf")
            if isinstance(entry, (tuple, list)) and len(entry) >= 2:
                try:
                    reward = float(entry[1])
                except Exception:
                    reward = float("-inf")
            scored_indices.append((reward, idx))

        scored_indices.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best_indices = {idx for _, idx in scored_indices[:best_keep]}

        selected_indices = sorted(recent_indices | best_indices)
        return [entries[idx] for idx in selected_indices]

    def _prune_replay_buffer_for_groq(self):
        """Preserve full replay history; prompt selection handles provider budgets."""
        return

    def _extract_reward_sequence(self, replay_entries):
        rewards = []
        for entry in replay_entries:
            if isinstance(entry, (tuple, list)) and len(entry) >= 2:
                try:
                    rewards.append(float(entry[1]))
                except Exception:
                    continue
        return rewards

    def _latest_recorded_reward(self):
        if len(self.replay_buffer.buffer) == 0:
            return None

        latest_entry = self.replay_buffer.buffer[-1]
        if not isinstance(latest_entry, (tuple, list)) or len(latest_entry) < 2:
            return None

        try:
            return float(latest_entry[1])
        except Exception:
            return None

    def _should_force_new_matrix_exploration(self):
        latest_reward = self._latest_recorded_reward()
        return latest_reward is not None and latest_reward < self.force_new_matrix_reward_threshold

    def _flatten_editable_values(self, payload, effective_frozen_factor=None):
        if isinstance(payload, dict):
            flat_parts = []
            for factor_name in self._factor_names():
                if factor_name == effective_frozen_factor:
                    continue
                if factor_name not in payload:
                    return None
                flat_parts.append(np.asarray(payload[factor_name], dtype=float).reshape(-1))

            if not flat_parts:
                return np.array([], dtype=float)
            return np.concatenate(flat_parts)

        vector = np.asarray(payload, dtype=float).reshape(-1)
        # For full-parameter policies, compare matrix indices only (exclude bias tail).
        if self.bias and vector.size >= self.dim_action:
            vector = vector[:-self.dim_action]
        return vector

    def _validate_forced_exploration_candidate(self, candidate_payload, effective_frozen_factor=None):
        candidate_vector = self._flatten_editable_values(candidate_payload, effective_frozen_factor)
        if candidate_vector is None:
            return False, "candidate payload is missing editable factor values"
        if candidate_vector.size == 0:
            return False, "candidate has no editable indices to validate"

        reference_entries = list(self.replay_buffer.buffer)[-self.force_exploration_reference_count:]
        if len(reference_entries) == 0:
            return True, "no references available"

        threshold = float(self.force_exploration_min_index_delta)

        for ref_pos, entry in enumerate(reference_entries, start=1):
            if not isinstance(entry, (tuple, list)) or len(entry) < 1:
                continue

            reference_payload = entry[0]
            reference_vector = self._flatten_editable_values(reference_payload, effective_frozen_factor)
            if reference_vector is None or reference_vector.size != candidate_vector.size:
                continue

            abs_diff = np.abs(candidate_vector - reference_vector)
            close_indices = np.where(abs_diff < threshold)[0]
            if close_indices.size > 0:
                first_index = int(close_indices[0])
                return (
                    False,
                    f"index {first_index} delta={abs_diff[first_index]:.4f} < "
                    f"required {threshold:.4f} against recent reference #{ref_pos}",
                )

        return True, "candidate passed strict index-wise uniqueness"

    def _build_reward_delta_context(self, replay_buffer: EpisodeRewardBufferNoBias):
        """Build prompt context for latest reward deltas.

        Deltas are computed from the full replay history so they reflect
        all previous matrix proposals and rewards.
        """
        rewards = self._extract_reward_sequence(list(replay_buffer.buffer))

        context = {
            "last_reward": None,
            "delta_from_prev_reward": None,
            "delta_from_best_reward": None,
            "delta_from_zero_reward": None,
            "distance_below_zero": None,
            "delta_toward_zero_from_prev": None,
        }
        if not rewards:
            return context

        last_reward = rewards[-1]
        best_reward = max(rewards)
        context["last_reward"] = f"{last_reward:.2f}"
        context["delta_from_zero_reward"] = f"{last_reward:+.2f}"
        context["distance_below_zero"] = f"{max(0.0, -last_reward):.2f}"
        if len(rewards) >= 2:
            prev_reward = rewards[-2]
            context["delta_from_prev_reward"] = f"{(last_reward - prev_reward):+.2f}"
            prev_below_zero_gap = max(0.0, -prev_reward)
            curr_below_zero_gap = max(0.0, -last_reward)
            context["delta_toward_zero_from_prev"] = f"{(prev_below_zero_gap - curr_below_zero_gap):+.2f}"
        context["delta_from_best_reward"] = f"{(last_reward - best_reward):+.2f}"
        return context

    def _dominant_value_ratio(self, vector, decimals=2):
        vector = np.asarray(vector, dtype=float)
        if vector.size == 0:
            return 0.0
        rounded = np.round(vector, decimals=decimals)
        _, counts = np.unique(rounded, return_counts=True)
        return float(np.max(counts) / rounded.size)

    def _cosine_similarity(self, vec_a, vec_b):
        vec_a = np.asarray(vec_a, dtype=float)
        vec_b = np.asarray(vec_b, dtype=float)
        norm_product = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        if norm_product < 1e-8:
            return 1.0 if np.linalg.norm(vec_a - vec_b) < 1e-8 else 0.0
        similarity = float(np.dot(vec_a, vec_b) / norm_product)
        return float(np.clip(similarity, -1.0, 1.0))

    def _assess_matrix_structure(self, matrix, matrix_name):
        """Assess matrix structure with hard invalidation and soft penalties."""
        matrix = np.asarray(matrix, dtype=float)
        hard_issues = []
        soft_notes = []
        penalty_score = 0.0

        def _dominance_threshold(length):
            if length >= 6:
                return 0.75
            if length >= 4:
                return 0.80
            if length == 3:
                return 1.00
            return 1.10  # effectively disabled for vectors shorter than 3

        def _inspect_axis(vectors, axis_name):
            nonlocal penalty_score
            vector_count = len(vectors)

            for idx, vec in enumerate(vectors):
                vec = np.asarray(vec, dtype=float)
                if vec.size >= 3:
                    dominant_ratio = self._dominant_value_ratio(vec)
                    threshold = _dominance_threshold(vec.size)
                    if dominant_ratio >= threshold:
                        hard_issues.append(
                            f"{matrix_name} {axis_name} {idx} is too repetitive "
                            f"(dominant value ratio {dominant_ratio:.2f}, threshold {threshold:.2f})."
                        )

                vec_scale = max(1.0, float(np.max(np.abs(vec))))
                vec_relative_std = float(np.std(vec) / vec_scale)
                if vec.size >= 3 and vec_relative_std < 0.08:
                    penalty_score += float((0.08 - vec_relative_std) * 2.0)
                    soft_notes.append(
                        f"{matrix_name} {axis_name} {idx} is close to flat "
                        f"(relative std {vec_relative_std:.3f})."
                    )

            for i in range(vector_count):
                vec_i = np.asarray(vectors[i], dtype=float)
                for j in range(i + 1, vector_count):
                    vec_j = np.asarray(vectors[j], dtype=float)
                    mean_abs_diff = float(np.mean(np.abs(vec_i - vec_j)))
                    cosine = self._cosine_similarity(vec_i, vec_j)

                    if np.allclose(vec_i, vec_j, atol=0.08, rtol=0.0):
                        hard_issues.append(
                            f"{matrix_name} {axis_name}s {i} and {j} are repeated or near-repeated."
                        )
                    elif cosine >= 0.995 and mean_abs_diff <= 0.20:
                        hard_issues.append(
                            f"{matrix_name} {axis_name}s {i} and {j} are too similar "
                            f"(cos {cosine:.3f}, mean abs diff {mean_abs_diff:.3f})."
                        )
                    elif cosine >= 0.96 and mean_abs_diff <= 0.35:
                        penalty_score += float((cosine - 0.96) * 4.0)
                        soft_notes.append(
                            f"{matrix_name} {axis_name}s {i} and {j} show repeating structure "
                            f"(cos {cosine:.3f})."
                        )

        row_vectors = [matrix[idx, :] for idx in range(matrix.shape[0])]
        col_vectors = [matrix[:, idx] for idx in range(matrix.shape[1])]
        _inspect_axis(row_vectors, "row")
        _inspect_axis(col_vectors, "column")

        if matrix.shape[0] >= 2:
            zero_tolerance = 0.05
            row_sign_patterns = []
            for row in row_vectors:
                signs = np.where(row > zero_tolerance, 1, np.where(row < -zero_tolerance, -1, 0))
                row_sign_patterns.append(tuple(signs.tolist()))

            unique_patterns = len(set(row_sign_patterns))
            pattern_diversity = unique_patterns / float(len(row_sign_patterns))
            if pattern_diversity < 0.50:
                penalty_score += float((0.50 - pattern_diversity) * 4.0)
                soft_notes.append(
                    f"{matrix_name} row sign patterns lack diversity "
                    f"({unique_patterns}/{len(row_sign_patterns)} unique)."
                )

        global_scale = max(1.0, float(np.max(np.abs(matrix))))
        global_relative_std = float(np.std(matrix) / global_scale)
        if global_relative_std < 0.12:
            penalty_score += float((0.12 - global_relative_std) * 6.0)
            soft_notes.append(
                f"{matrix_name} has low global variance (relative std {global_relative_std:.3f})."
            )

        if matrix.shape[0] >= 2:
            row_relative_std = np.std(matrix, axis=1) / global_scale
            flat_row_ratio = float(np.mean(row_relative_std < 0.08))
            if flat_row_ratio >= 0.50:
                penalty_score += float((flat_row_ratio - 0.50) * 5.0)
                soft_notes.append(
                    f"{matrix_name} has many flat rows ({flat_row_ratio:.0%})."
                )

        if matrix.shape[1] >= 2:
            col_relative_std = np.std(matrix, axis=0) / global_scale
            flat_col_ratio = float(np.mean(col_relative_std < 0.08))
            if flat_col_ratio >= 0.50:
                penalty_score += float((flat_col_ratio - 0.50) * 5.0)
                soft_notes.append(
                    f"{matrix_name} has many flat columns ({flat_col_ratio:.0%})."
                )

        if matrix.shape[0] == matrix.shape[1] and matrix.shape[0] >= 2:
            denom = float(np.mean(np.abs(matrix))) + 1e-8
            asymmetry = float(np.mean(np.abs(matrix - matrix.T)) / denom)
            symmetry_score = max(0.0, 1.0 - asymmetry)
            if symmetry_score >= 0.92:
                penalty_score += float((symmetry_score - 0.92) * 6.0)
                soft_notes.append(
                    f"{matrix_name} is close to symmetric (score {symmetry_score:.3f})."
                )

        # Keep logs compact and deterministic.
        hard_issues = list(dict.fromkeys(hard_issues))
        soft_notes = list(dict.fromkeys(soft_notes))
        if len(hard_issues) > 8:
            hard_issues = hard_issues[:8]
        if len(soft_notes) > 10:
            soft_notes = soft_notes[:10]

        return {
            "is_valid": len(hard_issues) == 0,
            "hard_issues": hard_issues,
            "penalty_score": float(min(12.0, max(0.0, penalty_score))),
            "soft_notes": soft_notes,
        }

    def _scale_structure_penalty(self, penalty_score, raw_reward):
        if penalty_score <= 0:
            return 0.0
        try:
            optimum_abs = abs(float(self.optimum))
        except Exception:
            optimum_abs = 0.0

        # Keep penalties meaningful across environments with different reward scales.
        reward_scale = max(1.0, optimum_abs / 200.0)
        scaled_penalty = float(penalty_score) * reward_scale

        # Avoid overwhelming the true environment signal.
        raw_scale = max(1.0, abs(float(raw_reward)))
        return float(min(scaled_penalty, raw_scale * 0.25))

    def _clone_factor_components(self, factor_components):
        if not isinstance(factor_components, dict):
            return None

        cloned = {}
        for factor_name in self._factor_names():
            if factor_name in factor_components:
                cloned[factor_name] = np.array(factor_components[factor_name], copy=True)

        if "bias" in factor_components:
            cloned["bias"] = np.array(factor_components["bias"], copy=True)

        return cloned

    def _apply_phase_best_on_switch(self, current_phase):
        """If phase changed, freeze the best matrix from the previous phase."""
        if self._lu_schedule_prev_phase == current_phase:
            return

        previous_phase = self._lu_schedule_prev_phase
        if previous_phase is not None and self._lu_phase_best_components is not None:
            if previous_phase == "update_L_freeze_U":
                # Previous phase optimized L; freeze the best L when moving to U updates.
                self.policy.factors['L'] = np.array(self._lu_phase_best_components['L'], copy=True)
                print(
                    "[LU Schedule] Phase switch: freezing best L from previous phase "
                    f"(best_reward={self._lu_phase_best_reward:.2f})."
                )
            elif previous_phase == "update_U_freeze_L":
                # Previous phase optimized U; freeze the best U when moving to L updates.
                self.policy.factors['U'] = np.array(self._lu_phase_best_components['U'], copy=True)
                print(
                    "[LU Schedule] Phase switch: freezing best U from previous phase "
                    f"(best_reward={self._lu_phase_best_reward:.2f})."
                )

            self.policy.reconstruct_weight_from_factors()

        self._lu_schedule_prev_phase = current_phase
        self._lu_phase_best_reward = float('-inf')
        self._lu_phase_best_components = None

    def _update_phase_best_components(self, phase_name, reward, factor_components):
        """Track the best-performing factor components in the current phase."""
        if self._lu_schedule_prev_phase != phase_name:
            return
        if not isinstance(factor_components, dict):
            return

        if reward > self._lu_phase_best_reward:
            self._lu_phase_best_reward = reward
            self._lu_phase_best_components = self._clone_factor_components(factor_components)
            print(
                "[LU Schedule] New phase best: "
                f"phase={phase_name}, reward={reward:.2f}"
            )

    def rollout_episode(self, world: BaseWorld, logging_file, record=True):
        # Ensure deterministic behavior for this episode
        self._set_global_seed(self.seed)
        state = world.reset(seed=self.seed)
        state = np.expand_dims(state, axis=0)
        
        # Get parameters for logging
        params = self.policy.get_parameters()
        if isinstance(params, dict):
            # For factorized policy, log the reconstructed weight
            logging_file.write(
                f"Weight matrix ({self.policy.get_factor_equation()}):\n{self.policy.weight}\n"
            )
            logging_file.write(f"Bias: {self.policy.bias}\n")
        else:
            logging_file.write(
                f"{', '.join([str(x) for x in params.reshape(-1)])}\n"
            )
        logging_file.write(f"parameter ends\n\n")
        logging_file.write(f"state | action | reward\n")
        done = False
        step_idx = 0
        while not done:
            action = self.policy.get_action(state.T)
            action = np.reshape(action, (1, self.dim_action))
            if world.discretize:
                action = np.argmax(action)
                action = np.array([action])
            next_state, reward, done = world.step(action)
            logging_file.write(f"{state.T[0]} | {action[0]} | {reward}\n")
            state = next_state
            step_idx += 1
            self.total_steps += 1
        logging_file.write(f"Total reward: {world.get_accu_reward()}\n")
        self.total_episodes += 1
        if record:
            self.replay_buffer.add(
                self.policy.get_parameters(), world.get_accu_reward()
            )
        return world.get_accu_reward()

    def record_best_episode(self, world: BaseWorld, logdir):
        """Record a video of the best performing policy."""
        try:
            # Create videos directory
            video_dir = f"{logdir}/best_videos"
            os.makedirs(video_dir, exist_ok=True)
            
            # Create a new environment with video recording wrapper
            env = gym.make(world.gym_env_name, render_mode="rgb_array")
            env = RecordVideo(
                env, 
                video_dir,
                episode_trigger=lambda x: True,  # Record every episode
                name_prefix=f"best_episode_{self.training_episodes}_reward_{self.best_reward:.0f}"
            )
            
            # Run one episode with the current policy
            # Seed the recording environment for deterministic playback
            if hasattr(env, 'reset'):
                try:
                    state, _ = env.reset(seed=self.seed)
                except TypeError:
                    state, _ = env.reset()
            else:
                state, _ = env.reset()
            state = np.expand_dims(state, axis=0)
            done = False
            total_reward = 0
            
            while not done:
                action = self.policy.get_action(state.T)
                action = np.reshape(action, (1, self.dim_action))
                if world.discretize:
                    action = np.argmax(action)
                    action = np.array([action])
                
                next_state, reward, terminated, truncated, _ = env.step(action[0])
                total_reward += reward
                done = terminated or truncated
                state = np.expand_dims(next_state, axis=0)
            
            env.close()
            print(f"✓ Video saved to {video_dir}/ (reward: {total_reward:.2f})")
            
        except Exception as e:
            print(f"Warning: Could not record video: {e}")

    def _set_global_seed(self, seed: int):
        """Set seeds for Python, NumPy, Torch (if available)."""
        try:
            random.seed(seed)
        except Exception:
            pass
        try:
            np.random.seed(seed)
        except Exception:
            pass
        if TORCH_AVAILABLE:
            try:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            except Exception:
                pass

    def random_warmup(self, world: BaseWorld, logdir, num_episodes):
        for episode in range(num_episodes):
            self.policy.initialize_policy()
            # Run the episode and collect the trajectory
            print(f"Rolling out warmup episode {episode}...")
            logging_filename = f"{logdir}/warmup_rollout_{episode}.txt"
            logging_file = open(logging_filename, "w", encoding="utf-8")
            result = self.rollout_episode(world, logging_file)
            print(f"Result: {result}")
            
            # Track warmup rewards
            self.warmup_rewards.append(result)
            
            # Save heatmap for warmup episodes
            if episode == num_episodes - 1:  # Save visualization after last warmup
                self.plot_reward_progress(logdir)
                self.plot_policy_heatmap(logdir)

    def train_policy(self, world: BaseWorld, logdir):

        effective_frozen_factor = self.frozen_factor
        force_new_matrix_exploration = self._should_force_new_matrix_exploration()
        if force_new_matrix_exploration:
            latest_reward = self._latest_recorded_reward()
            print(
                "[Exploration Reset] "
                f"latest reward={latest_reward:.2f} < {self.force_new_matrix_reward_threshold:.2f}. "
                "Disabling exploitation and requesting a completely new matrix proposal."
            )

        def parse_parameters(input_text):
            # This regex looks for integers or floating-point numbers (including optional sign)
            s = input_text.split("\n")[0]
            print("response:", s)
            pattern = re.compile(r"params\[(\d+)\]:\s*([+-]?\d+(?:\.\d+)?)")
            matches = pattern.findall(s)

            # Convert matched strings to float (or int if you prefer to differentiate)
            results = []
            for match in matches:
                results.append(float(match[1]))
            print(results)
            assert len(results) == self.rank
            return np.array(results).reshape(-1)
        
        def parse_factor_matrices(input_text):
            """Parse decomposition factors from LLM output."""
            self._current_matrix_structure_penalty = 0.0
            self._current_matrix_quality_notes = []

            factor_names = self._factor_names()
            factor_shapes = self._factor_shapes()
            current_components = self.policy.get_parameters(return_factors=True)

            fallback_components = {
                factor_name: np.array(current_components[factor_name], copy=True)
                for factor_name in factor_names
            }
            fallback_components["bias"] = np.array(self.policy.bias, copy=True)

            lines = input_text.strip().split('\n')

            def detect_section(raw_line):
                """Detect factor/bias section headings with flexible markdown support."""
                if not raw_line:
                    return None

                normalized = raw_line.strip().lower()
                normalized = re.sub(r'^[#>*\-\s]+', '', normalized)
                normalized = normalized.strip('`*_ ')

                heading_prefix = r'(?:optimized|updated|new|candidate|final|proposed|fixed|frozen)?\s*'
                heading_suffix = r'(?:\s*\([^\)]*\))?\s*[:=-]?\s*$'

                if re.match(r'^' + heading_prefix + r'bias(?:\s+vector)?' + heading_suffix, normalized):
                    return 'bias'

                # Try to extract a factor token from the heading and canonicalize it.
                heading_match = re.match(
                    r'^' + heading_prefix + r'([a-zA-Z^\s]+?)' + heading_suffix,
                    normalized,
                )
                if heading_match:
                    candidate_name = self._canonicalize_factor_name(heading_match.group(1))
                    if candidate_name in factor_names:
                        return candidate_name

                # Fallback: direct canonicalization of the whole line.
                candidate_name = self._canonicalize_factor_name(normalized)
                if candidate_name in factor_names:
                    return candidate_name

                return None

            parsed_rows = {factor_name: [] for factor_name in factor_names}
            parsed_vectors = {factor_name: [] for factor_name in factor_names}
            bias_vector = []
            current_section = None

            for line in lines:
                line = line.strip()
                detected_section = detect_section(line)
                if detected_section is not None:
                    current_section = detected_section
                    continue
                elif 'Explanation:' in line or 'explanation:' in line or line.startswith('Note:'):
                    break

                # Parse numerical values
                if current_section and line and not line.startswith('Explanation') and not line.startswith('Note'):
                    # Extract numbers from the line (including negative numbers and decimals)
                    numbers = re.findall(r'[+-]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][+-]?\d+)?', line)
                    if numbers:
                        row = [float(x) for x in numbers]

                        if current_section == 'bias':
                            bias_vector.extend(row)
                            continue

                        if current_section == effective_frozen_factor:
                            continue

                        expected_shape = factor_shapes[current_section]
                        if len(expected_shape) == 1:
                            parsed_vectors[current_section].extend(row)
                        else:
                            expected_cols = expected_shape[1]
                            if len(row) == expected_cols:
                                parsed_rows[current_section].append(row)
                            else:
                                print(
                                    f"Warning: Skipping {current_section} row with {len(row)} values "
                                    f"(expected {expected_cols}): {row}"
                                )

            parsed_components = {}
            for factor_name in factor_names:
                if factor_name == effective_frozen_factor:
                    parsed_components[factor_name] = np.array(current_components[factor_name], copy=True)
                    continue

                expected_shape = factor_shapes[factor_name]

                if len(expected_shape) == 1:
                    factor_value = np.array(parsed_vectors[factor_name], dtype=float)
                    if factor_value.shape != expected_shape:
                        print(
                            f"ERROR: {factor_name} vector has shape {factor_value.shape}, "
                            f"expected {expected_shape}"
                        )
                        return fallback_components
                    parsed_components[factor_name] = factor_value
                else:
                    try:
                        factor_value = np.array(parsed_rows[factor_name], dtype=float)
                    except ValueError as exc:
                        print(f"ERROR creating factor {factor_name}: {exc}")
                        return fallback_components

                    if factor_value.shape != expected_shape:
                        if factor_name == "Vt" and factor_value.ndim == 2 and factor_value.T.shape == expected_shape:
                            factor_value = factor_value.T
                        else:
                            print(
                                f"ERROR: {factor_name} matrix has shape {factor_value.shape}, "
                                f"expected {expected_shape}"
                            )
                            return fallback_components
                    parsed_components[factor_name] = factor_value

            bias = np.array(bias_vector).reshape(1, -1) if bias_vector else np.array(self.policy.bias, copy=True)
            if bias.shape != self.policy.bias.shape:
                print(
                    f"ERROR: bias has shape {bias.shape}, expected {self.policy.bias.shape}. "
                    "Keeping previous factors."
                )
                return fallback_components

            try:
                normalized_factors = self.policy.normalize_factor_components(parsed_components)
            except ValueError as exc:
                print(f"ERROR: invalid factor proposal: {exc}")
                return fallback_components

            total_penalty_score = 0.0
            quality_notes = []
            structural_issues = []

            for factor_name in factor_names:
                if factor_name == effective_frozen_factor:
                    continue

                factor_value = np.asarray(normalized_factors[factor_name], dtype=float)
                if factor_value.ndim == 2:
                    assessment = self._assess_matrix_structure(factor_value, factor_name)
                    total_penalty_score += assessment["penalty_score"]
                    quality_notes.extend(assessment["soft_notes"])
                    structural_issues.extend(assessment["hard_issues"])
                else:
                    # Light regularity penalty for vectors (e.g., S in SVD).
                    scale = max(1.0, float(np.max(np.abs(factor_value))))
                    relative_std = float(np.std(factor_value) / scale) if factor_value.size > 1 else 0.0
                    if factor_value.size > 1 and relative_std < 0.05:
                        total_penalty_score += float((0.05 - relative_std) * 2.0)
                        quality_notes.append(
                            f"{factor_name} vector has low spread (relative std {relative_std:.3f})."
                        )

            if structural_issues:
                print("ERROR: Rejecting matrix proposal due to structural invalidation:")
                for issue in structural_issues:
                    print(f" - {issue}")
                return fallback_components

            self._current_matrix_structure_penalty = float(total_penalty_score)
            self._current_matrix_quality_notes = list(dict.fromkeys(quality_notes))

            if self._current_matrix_structure_penalty > 0:
                print(
                    "[Matrix Quality] Soft penalty score: "
                    f"{self._current_matrix_structure_penalty:.3f}"
                )
                for note in self._current_matrix_quality_notes:
                    print(f" - {note}")

            print("✓ Shapes validated correctly")
            for factor_name in factor_names:
                if factor_name != effective_frozen_factor:
                    print(f"{factor_name} shape: {normalized_factors[factor_name].shape}")

            parsed_payload = {
                factor_name: normalized_factors[factor_name]
                for factor_name in factor_names
            }
            parsed_payload["bias"] = bias
            return parsed_payload

        def str_nd_examples(replay_buffer: EpisodeRewardBufferNoBias, n):
            if force_new_matrix_exploration:
                return (
                    "Exploration reset mode is active because the latest reward dropped below -100.\n"
                    "Do NOT exploit or reuse prior parameter structures.\n"
                    "Propose a completely new parameter matrix with a distinctly different value layout.\n"
                )

            selected_entries = self._select_prompt_replay_entries(replay_buffer)
            all_parameters = []
            for weights, reward in selected_entries:
                parameters = weights
                all_parameters.append((parameters.reshape(-1), reward))

            text = ""
            if self._is_groq_mode() and len(selected_entries) > 0:
                text += "(Groq mode) Showing latest 4 attempts + top 3 historical rewards.\n"
            for parameters, reward in all_parameters:
                l = ""
                for i in range(n):
                    l += f"params[{i}]: {parameters[i]:.5g}; "
                fxy = reward
                l += f"f(params): {fxy:.2f}\n"
                text += l
            return text
        
        def str_factor_examples(replay_buffer: EpisodeRewardBufferNoBias):
            """Format examples showing decomposition factors and rewards."""
            factor_names = self._factor_names()

            def _format_factor_block(name, value):
                value = np.asarray(value)
                block = f"{name} factor:\n"
                if value.ndim == 1:
                    block += ", ".join([f"{x:.2f}" for x in value]) + "\n"
                else:
                    for row in value:
                        block += ", ".join([f"{x:.2f}" for x in row]) + "\n"
                return block

            # ---- Frozen-factor preamble ----
            preamble = ""
            if effective_frozen_factor is not None:
                current = self.policy.get_parameters(return_factors=True)
                if effective_frozen_factor in current:
                    preamble += (
                        f"FIXED {effective_frozen_factor} factor "
                        "(stays constant - do NOT change this):\n"
                    )
                    preamble += _format_factor_block(
                        effective_frozen_factor,
                        current[effective_frozen_factor],
                    )
                    preamble += "\n"

            if force_new_matrix_exploration:
                return (
                    preamble
                    + "Exploration reset mode is active because the latest reward dropped below -100.\n"
                    + "Do NOT exploit or reuse previous factor structures.\n"
                    + "Generate completely new values for all editable factors.\n"
                )

            selected_entries = self._select_prompt_replay_entries(replay_buffer)

            if len(selected_entries) == 0:
                return preamble + "(No previous attempts yet)\n"

            text = preamble
            if self._is_groq_mode():
                text += "(Groq mode) Showing latest 4 attempts + top 3 historical rewards.\n"
            text += f"Total previous attempts shown: {len(selected_entries)}\n"
            text += "=" * 60 + "\n\n"

            for idx, (weights, reward) in enumerate(selected_entries, 1):
                if isinstance(weights, dict) and all(name in weights for name in factor_names):
                    text += f"Attempt #{idx}:\n"

                    for factor_name in factor_names:
                        if factor_name == effective_frozen_factor:
                            continue
                        text += _format_factor_block(factor_name, weights[factor_name])

                    text += f"f(params): {reward:.2f}\n\n"
                else:
                    # Fallback to flat parameters
                    parameters = weights.reshape(-1)
                    text += f"Attempt #{idx}:\n"
                    text += "params: " + ", ".join([f"{x:.2g}" for x in parameters[:10]]) + "...\n"
                    text += f"f(params): {reward:.2f}\n\n"

            text += "=" * 60 + "\n"
            return text

        # Update the policy using llm_brain, q_table and replay_buffer
        self._prune_replay_buffer_for_groq()
        print("Updating the policy...")

        current_frozen_factor = self.frozen_factor
        schedule_context = None
        schedule_phase = None
        if (
            self.use_factorized_policy
            and self.decomposition_type == "lu"
            and self.enable_alternating_lu_schedule
        ):
            schedule_period = self.lu_schedule_l_episodes + self.lu_schedule_u_iterations
            schedule_step = self.training_episodes % schedule_period
            if schedule_step < self.lu_schedule_l_episodes:
                # L update phase: freeze U, optimize L.
                current_frozen_factor = 'U'
                schedule_phase = "update_L_freeze_U"
            else:
                # U update phase: freeze L, optimize U.
                current_frozen_factor = 'L'
                schedule_phase = "update_U_freeze_L"

            schedule_context = {
                "enabled": True,
                "phase": schedule_phase,
                "l_episodes": self.lu_schedule_l_episodes,
                "u_iterations": self.lu_schedule_u_iterations,
                "cycle_step": schedule_step + 1,
                "cycle_length": schedule_period,
            }
            print(
                "[LU Schedule] "
                f"cycle_step={schedule_step + 1}/{schedule_period}, "
                f"phase={schedule_phase}, frozen_factor={current_frozen_factor}"
            )

            # On phase boundaries, freeze the best matrix from the previous phase.
            self._apply_phase_best_on_switch(schedule_phase)

        effective_frozen_factor = current_frozen_factor
        reward_delta_context = self._build_reward_delta_context(self.replay_buffer)
        reward_delta_context["force_new_matrix_exploration"] = force_new_matrix_exploration
        reward_delta_context["force_new_matrix_threshold"] = self.force_new_matrix_reward_threshold
        reward_delta_context["force_new_matrix_index_delta"] = self.force_exploration_min_index_delta
        reward_delta_context["force_new_matrix_reference_count"] = self.force_exploration_reference_count
        
        if self.use_factorized_policy:
            self._current_matrix_structure_penalty = 0.0
            self._current_matrix_quality_notes = []

            # Factorized policy: LLM generates decomposition factors, then weight is reconstructed.
            llm_attempt_budget = (
                self.force_exploration_max_llm_attempts
                if force_new_matrix_exploration
                else 1
            )
            new_factor_components = None
            reasoning = None
            last_validation_reason = None

            for llm_attempt_idx in range(llm_attempt_budget):
                candidate_components, candidate_reasoning, api_time = self.llm_brain.llm_update_parameters_num_optim(
                    str_factor_examples(self.replay_buffer),
                    parse_factor_matrices,
                    self.training_episodes,
                    self.rank,
                    self.optimum,
                    self.search_step_size,
                    dim_state=self.policy.dim_states,
                    dim_action=self.policy.dim_actions,
                    factor_rank=self.factor_rank,
                    use_factorized=True,
                    decomposition_type=self.decomposition_type,
                    factor_names=self._factor_names(),
                    frozen_factor=current_frozen_factor,
                    schedule_context=schedule_context,
                    reward_context=reward_delta_context,
                )
                self.api_call_time += api_time

                if force_new_matrix_exploration:
                    is_valid, validation_reason = self._validate_forced_exploration_candidate(
                        candidate_components,
                        effective_frozen_factor=effective_frozen_factor,
                    )
                    if not is_valid:
                        last_validation_reason = validation_reason
                        print(
                            "[Exploration Reset] Rejected factor proposal "
                            f"attempt {llm_attempt_idx + 1}/{llm_attempt_budget}: {validation_reason}"
                        )
                        continue

                new_factor_components = candidate_components
                reasoning = candidate_reasoning
                break

            if new_factor_components is None:
                raise ValueError(
                    "Forced exploration failed to produce a strictly unique factor proposal. "
                    f"Last rejection: {last_validation_reason}"
                )

            factor_shape_log = {
                factor_name: np.asarray(new_factor_components[factor_name]).shape
                for factor_name in self._factor_names()
                if factor_name in new_factor_components
            }
            print(f"Factor shapes: {factor_shape_log}")
            self.policy.update_policy(factor_components=new_factor_components)
            print(f"Weight shape after update: {self.policy.weight.shape}")
            
            # Store factor components for replay buffer
            new_parameter_list = new_factor_components
        else:
            # Use regular parameter optimization
            llm_attempt_budget = (
                self.force_exploration_max_llm_attempts
                if force_new_matrix_exploration
                else 1
            )
            new_parameter_list = None
            reasoning = None
            last_validation_reason = None

            for llm_attempt_idx in range(llm_attempt_budget):
                candidate_parameters, candidate_reasoning, api_time = self.llm_brain.llm_update_parameters_num_optim(
                    str_nd_examples(self.replay_buffer, self.rank),
                    parse_parameters,
                    self.training_episodes,
                    self.rank,
                    self.optimum,
                    self.search_step_size,
                    reward_context=reward_delta_context,
                )
                self.api_call_time += api_time

                if force_new_matrix_exploration:
                    is_valid, validation_reason = self._validate_forced_exploration_candidate(
                        candidate_parameters,
                        effective_frozen_factor=None,
                    )
                    if not is_valid:
                        last_validation_reason = validation_reason
                        print(
                            "[Exploration Reset] Rejected parameter proposal "
                            f"attempt {llm_attempt_idx + 1}/{llm_attempt_budget}: {validation_reason}"
                        )
                        continue

                new_parameter_list = candidate_parameters
                reasoning = candidate_reasoning
                break

            if new_parameter_list is None:
                raise ValueError(
                    "Forced exploration failed to produce a strictly unique parameter proposal. "
                    f"Last rejection: {last_validation_reason}"
                )

            print(self.policy.get_parameters().shape)
            print(new_parameter_list.shape)
            self.policy.update_policy(new_parameter_list)
            print(self.policy.get_parameters().shape)
        
        logging_q_filename = f"{logdir}/parameters.txt"
        logging_q_file = open(logging_q_filename, "w", encoding="utf-8")
        logging_q_file.write(str(self.policy))
        logging_q_file.close()
        q_reasoning_filename = f"{logdir}/parameters_reasoning.txt"
        q_reasoning_file = open(q_reasoning_filename, "w", encoding="utf-8")
        q_reasoning_file.write(reasoning)
        q_reasoning_file.close()
        print("Policy updated!")

        # Run the episode and collect the trajectory
        print(f"Rolling out episode {self.training_episodes}...")
        logging_filename = f"{logdir}/training_rollout.txt"
        logging_file = open(logging_filename, "w", encoding="utf-8")
        results = []
        for idx in range(self.num_evaluation_episodes):
            if idx == 0:
                result = self.rollout_episode(world, logging_file, record=False)
            else:
                result = self.rollout_episode(world, logging_file, record=False)
            results.append(result)
        print(f"Results: {results}")
        result = np.mean(results)
        variance = np.var(results)
        std = np.std(results)
        print(f"Mean: {result:.2f}, Variance: {variance:.2f}, Std: {std:.2f}")

        if self.use_factorized_policy:
            raw_penalty_score = float(getattr(self, "_current_matrix_structure_penalty", 0.0))
            if raw_penalty_score > 0:
                print(
                    "[Matrix Quality] Soft-penalty signal (informational only): "
                    f"raw_score={raw_penalty_score:.3f}; reward buffer keeps raw reward={result:.2f}"
                )
                for note in getattr(self, "_current_matrix_quality_notes", []):
                    print(f" - {note}")

        if (
            self.use_factorized_policy
            and self.decomposition_type == "lu"
            and self.enable_alternating_lu_schedule
            and schedule_phase is not None
        ):
            self._update_phase_best_components(schedule_phase, result, new_parameter_list)

        self.replay_buffer.add(new_parameter_list, result)
        self._prune_replay_buffer_for_groq()
        
        # Track training rewards only
        self.training_rewards.append(result)
        
        # Record video if this is a new best reward
        if result > self.best_reward:
            print(f"\n🎉 New best reward! {result:.2f} > {self.best_reward:.2f}")
            print(f"Recording video of best performance...")
            self.best_reward = result
            self.record_best_episode(world, logdir)
        
        # Create visualizations iteratively after each LLM call
        print(f"\n[Visualization] Generating plots for episode {self.training_episodes}...")
        self.plot_reward_progress(logdir)
        self.plot_policy_heatmap(logdir)
        print(f"[Visualization] Plots saved to {logdir}")

        self.training_episodes += 1

        _cpu_time = time.process_time() - self.start_time
        _api_time = self.api_call_time
        _total_episodes = self.total_episodes
        _total_steps = self.total_steps
        _total_reward = result
        _variance = variance
        _std = std
        return _cpu_time, _api_time, _total_episodes, _total_steps, _total_reward, _variance, _std
    
    def plot_reward_progress(self, logdir):
        """Plot training episode rewards (excluding warmup)."""
        if len(self.training_rewards) == 0:
            return  # No training rewards to plot yet
            
        plt.figure(figsize=(10, 6))
        episodes = list(range(len(self.training_rewards)))
        
        plt.plot(episodes, self.training_rewards, 'b-', marker='o', markersize=4, linewidth=2)
        # Draw best reward indicator (horizontal line + marker)
        try:
            # Determine best reward from tracked value and current rewards
            candidate_best = [r for r in self.training_rewards if r is not None]
            if self.best_reward is not None and self.best_reward != -float('inf'):
                candidate_best.append(self.best_reward)
            best_value = max(candidate_best) if candidate_best else None
        except Exception:
            best_value = None

        if best_value is not None:
            plt.axhline(best_value, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Best: {best_value:.2f}')
            # Mark the episode where the best reward occurred (from history)
            try:
                best_idx = int(np.argmax(self.training_rewards))
                plt.scatter([best_idx], [self.training_rewards[best_idx]], color='red', s=80, zorder=5)
            except Exception:
                pass

        plt.xlabel('Training Episode', fontsize=12)
        plt.ylabel('Reward', fontsize=12)
        plt.title('Training Reward Progress', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        plot_filename = f"{logdir}/reward_progress.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved reward progress plot to {plot_filename}")
    
    def plot_policy_heatmap(self, logdir):
        """Plot heatmap of the current policy weight matrix. Generated iteratively after each LLM call."""
        plt.figure(figsize=(10, 8))
        
        # Get the weight matrix
        weight_matrix = self.policy.weight
        
        # Create heatmap
        sns.heatmap(
            weight_matrix,
            annot=True,
            fmt='.2f',
            cmap='viridis',
            cbar_kws={'label': 'Weight Value'},
            linewidths=0.5,
            linecolor='gray'
        )
        
        plt.xlabel('Action Dimension', fontsize=12)
        plt.ylabel('State Dimension', fontsize=12)
        plt.title(f'Policy Weight Matrix - Episode {self.training_episodes}', fontsize=14)
        
        plot_filename = f"{logdir}/policy_heatmap_ep{self.training_episodes}.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved policy heatmap to {plot_filename}")
        
        # If using factorized policy, also save decomposition factor heatmaps.
        if self.use_factorized_policy and self.policy.get_factor_names():
            factor_components = self.policy.get_parameters(return_factors=True)
            factor_names = self.policy.get_factor_names()
            subplot_count = len(factor_names) + 1

            fig, axes = plt.subplots(1, subplot_count, figsize=(6 * subplot_count, 5))
            if subplot_count == 2:
                axes = [axes[0], axes[1]]

            for idx, factor_name in enumerate(factor_names):
                factor_value = np.asarray(factor_components[factor_name], dtype=float)
                if factor_value.ndim == 1:
                    factor_plot = factor_value.reshape(1, -1)
                    y_label = 'Component'
                else:
                    factor_plot = factor_value
                    y_label = 'Row'

                sns.heatmap(
                    factor_plot,
                    annot=True,
                    fmt='.2f',
                    cmap='viridis',
                    ax=axes[idx],
                    cbar_kws={'label': 'Value'}
                )
                axes[idx].set_title(f'{factor_name} Factor')
                axes[idx].set_xlabel('Column')
                axes[idx].set_ylabel(y_label)

            sns.heatmap(
                weight_matrix,
                annot=True,
                fmt='.2f',
                cmap='viridis',
                ax=axes[-1],
                cbar_kws={'label': 'Value'}
            )
            axes[-1].set_title(f"Policy Weight ({self.policy.get_factor_equation()})")
            axes[-1].set_xlabel('Action Dimension')
            axes[-1].set_ylabel('State Dimension')

            plt.suptitle(
                f"{self.decomposition_type.upper()} Factorized Policy - Episode {self.training_episodes}",
                fontsize=16,
            )
            plt.tight_layout()

            factor_plot_filename = f"{logdir}/policy_factor_heatmap_ep{self.training_episodes}.png"
            plt.savefig(factor_plot_filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved factorized policy heatmap to {factor_plot_filename}")
    
    def create_heatmap_gifs(self, logdir, duration=500, loop=0):
        """
        Create animated GIFs from the heatmap images generated during training.
        
        Args:
            logdir (str): Directory containing the heatmap images
            duration (int): Duration of each frame in milliseconds (default 500ms)
            loop (int): Number of times to loop (0 = infinite loop)
        
        Returns:
            tuple: Paths to (policy_gif, lu_gif) or None for failed ones
        """
        try:
            from PIL import Image
        except ImportError:
            print("Warning: PIL (Pillow) not installed. Cannot create GIFs.")
            print("Install with: pip install Pillow")
            return None, None
        
        from pathlib import Path
        
        def create_gif(heatmap_type, output_filename):
            """Helper function to create a single GIF."""
            logdir_path = Path(logdir)
            pattern = f"{heatmap_type}_ep(\\d+)\\.png"
            heatmap_files = []
            
            # Search in main directory
            for file in logdir_path.glob(f"{heatmap_type}_ep*.png"):
                match = re.search(pattern, file.name)
                if match:
                    episode_num = int(match.group(1))
                    heatmap_files.append((episode_num, file))
            
            # Search in episode subdirectories
            for episode_dir in logdir_path.glob("episode_*"):
                if episode_dir.is_dir():
                    for file in episode_dir.glob(f"{heatmap_type}_ep*.png"):
                        match = re.search(pattern, file.name)
                        if match:
                            episode_num = int(match.group(1))
                            heatmap_files.append((episode_num, file))
            
            if not heatmap_files:
                print(f"Warning: No {heatmap_type} images found in {logdir} or its episode subdirectories")
                return None
            
            # Sort by episode number
            heatmap_files.sort(key=lambda x: x[0])
            
            print(f"Found {len(heatmap_files)} {heatmap_type} images")
            print(f"Episode range: {heatmap_files[0][0]} to {heatmap_files[-1][0]}")
            
            # Load images
            images = []
            for episode_num, filepath in heatmap_files:
                try:
                    img = Image.open(filepath)
                    images.append(img)
                except Exception as e:
                    print(f"Warning: Could not load {filepath}: {e}")
            
            if not images:
                print(f"Error: No {heatmap_type} images could be loaded")
                return None
            
            # Save as GIF
            output_path = logdir_path / output_filename
            
            try:
                images[0].save(
                    output_path,
                    save_all=True,
                    append_images=images[1:],
                    duration=duration,
                    loop=loop,
                    optimize=False
                )
                print(f"Successfully created GIF: {output_path}")
                print(f"Total frames: {len(images)}, Frame duration: {duration}ms")
                return str(output_path)
            except Exception as e:
                print(f"Error creating {heatmap_type} GIF: {e}")
                return None
        
        print("\n" + "=" * 60)
        print("Creating Animated GIFs from Heatmaps...")
        print("=" * 60)
        
        # Create policy heatmap GIF
        print("\nCreating Policy Weight Heatmap GIF...")
        policy_gif = create_gif("policy_heatmap", "policy_heatmaps.gif")
        
        # Create factorized policy heatmap GIF if using factorized policy
        factor_gif = None
        if self.use_factorized_policy:
            print("\nCreating Factorized Policy (L, U) Heatmap GIF...")
            factor_gif = create_gif("policy_factor_heatmap", "policy_factor_heatmaps.gif")
        
        print("=" * 60)
        return policy_gif, factor_gif

    def evaluate_policy(self, world: BaseWorld, logdir):
        results = []
        for idx in range(self.num_evaluation_episodes):
            logging_filename = f"{logdir}/evaluation_rollout_{idx}.txt"
            logging_file = open(logging_filename, "w", encoding="utf-8")
            result = self.rollout_episode(world, logging_file, record=False)
            results.append(result)
        return results
