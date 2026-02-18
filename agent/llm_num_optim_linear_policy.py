from agent.policy.linear_policy_no_bias import LinearPolicy as LinearPolicyNoBias
from agent.policy.linear_policy import LinearPolicy
from agent.policy.replay_buffer import EpisodeRewardBufferNoBias
from agent.policy.llm_brain_linear_policy import LLMBrain
from world.base_world import BaseWorld
import numpy as np
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Enable headless rendering for video recording on servers without display
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
        use_lu_factorization=False,
        lu_rank=None,
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
        self.use_lu_factorization = use_lu_factorization

        if not self.bias:
            param_count = dim_action * dim_state
        else:
            param_count = dim_action * dim_state + dim_action
        self.rank = param_count
        
        # Setup LU rank
        if use_lu_factorization:
            if lu_rank is None:
                self.lu_rank = max(1, min(dim_state, dim_action) // 2)
            else:
                self.lu_rank = lu_rank
        else:
            self.lu_rank = None

        if not self.bias:
            self.policy = LinearPolicyNoBias(
                dim_actions=dim_action, dim_states=dim_state
            )
        else:
            self.policy = LinearPolicy(
                dim_actions=dim_action, 
                dim_states=dim_state,
                use_lu_factorization=use_lu_factorization,
                lu_rank=self.lu_rank
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

        if self.bias:
            self.dim_state += 1

    def rollout_episode(self, world: BaseWorld, logging_file, record=True):
        state = world.reset()
        state = np.expand_dims(state, axis=0)
        
        # Get parameters for logging
        params = self.policy.get_parameters()
        if isinstance(params, dict):
            # For LU factorization, log the reconstructed weight
            logging_file.write(f"Weight matrix (L @ U):\n{self.policy.weight}\n")
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

    def random_warmup(self, world: BaseWorld, logdir, num_episodes):
        for episode in range(num_episodes):
            self.policy.initialize_policy()
            # Run the episode and collect the trajectory
            print(f"Rolling out warmup episode {episode}...")
            logging_filename = f"{logdir}/warmup_rollout_{episode}.txt"
            logging_file = open(logging_filename, "w")
            result = self.rollout_episode(world, logging_file)
            print(f"Result: {result}")
            
            # Track warmup rewards
            self.warmup_rewards.append(result)
            
            # Save heatmap for warmup episodes
            if episode == num_episodes - 1:  # Save visualization after last warmup
                self.plot_reward_progress(logdir)
                self.plot_policy_heatmap(logdir)

    def train_policy(self, world: BaseWorld, logdir):

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
        
        def parse_lu_matrices(input_text):
            """Parse L and U matrices from LLM output."""
            lines = input_text.strip().split('\n')
            
            L_matrix = []
            U_matrix = []
            bias_vector = []
            
            current_section = None
            expected_L_cols = self.lu_rank
            expected_U_cols = self.policy.dim_actions
            
            for line in lines:
                line = line.strip()
                if 'L matrix:' in line or 'L Matrix:' in line:
                    current_section = 'L'
                    continue
                elif 'U matrix:' in line or 'U Matrix:' in line:
                    current_section = 'U'
                    continue
                elif 'Bias:' in line or 'bias:' in line:
                    current_section = 'bias'
                    continue
                elif 'Explanation:' in line or 'explanation:' in line or line.startswith('Note:'):
                    break
                
                # Parse numerical values
                if current_section and line and not line.startswith('Explanation') and not line.startswith('Note'):
                    # Extract numbers from the line (including negative numbers and decimals)
                    numbers = re.findall(r'[+-]?\d+(?:\.\d+)?', line)
                    if numbers:
                        row = [float(x) for x in numbers]
                        
                        # Validate row length before adding
                        if current_section == 'L':
                            if len(row) == expected_L_cols:
                                L_matrix.append(row)
                            else:
                                print(f"Warning: Skipping L row with {len(row)} values (expected {expected_L_cols}): {row}")
                        elif current_section == 'U':
                            if len(row) == expected_U_cols:
                                U_matrix.append(row)
                            else:
                                print(f"Warning: Skipping U row with {len(row)} values (expected {expected_U_cols}): {row}")
                        elif current_section == 'bias':
                            bias_vector.extend(row)
            
            print(f"Parsed {len(L_matrix)} L rows, {len(U_matrix)} U rows")
            
            # Convert to numpy arrays with validation
            try:
                L = np.array(L_matrix)
            except ValueError as e:
                print(f"ERROR creating L matrix: {e}")
                print(f"L_matrix content: {L_matrix}")
                return {'L': self.policy.L.copy(), 'U': self.policy.U_matrix.copy(), 'bias': self.policy.bias.copy()}
            
            try:
                U = np.array(U_matrix)
            except ValueError as e:
                print(f"ERROR creating U matrix: {e}")
                print(f"U_matrix content: {U_matrix}")
                return {'L': self.policy.L.copy(), 'U': self.policy.U_matrix.copy(), 'bias': self.policy.bias.copy()}
            
            bias = np.array(bias_vector).reshape(1, -1) if bias_vector else self.policy.bias
            
            print(f"Parsed L shape: {L.shape}, U shape: {U.shape}, bias shape: {bias.shape}")
            
            # Validate shapes
            expected_L_shape = (self.policy.dim_states, self.lu_rank)
            expected_U_shape = (self.lu_rank, self.policy.dim_actions)
            
            if L.shape != expected_L_shape:
                print(f"ERROR: L matrix has wrong shape {L.shape}, expected {expected_L_shape}")
                print(f"LLM provided {len(L_matrix)} rows, expected {expected_L_shape[0]} rows with {expected_L_shape[1]} columns each")
                # Return previous policy to continue training
                return {'L': self.policy.L.copy(), 'U': self.policy.U_matrix.copy(), 'bias': self.policy.bias.copy()}
            
            if U.shape != expected_U_shape:
                print(f"ERROR: U matrix has wrong shape {U.shape}, expected {expected_U_shape}")
                print(f"LLM provided {len(U_matrix)} rows, expected {expected_U_shape[0]} rows with {expected_U_shape[1]} columns each")
                # Return previous policy to continue training
                return {'L': self.policy.L.copy(), 'U': self.policy.U_matrix.copy(), 'bias': self.policy.bias.copy()}
            
            print(f"✓ Shapes validated correctly")
            print(f"L matrix:\n{L}")
            print(f"U matrix:\n{U}")
            
            return {'L': L, 'U': U, 'bias': bias}

        def str_nd_examples(replay_buffer: EpisodeRewardBufferNoBias, n):

            all_parameters = []
            for weights, reward in replay_buffer.buffer:
                parameters = weights
                all_parameters.append((parameters.reshape(-1), reward))

            text = ""
            for parameters, reward in all_parameters:
                l = ""
                for i in range(n):
                    l += f"params[{i}]: {parameters[i]:.5g}; "
                fxy = reward
                l += f"f(params): {fxy:.2f}\n"
                text += l
            return text
        
        def str_lu_examples(replay_buffer: EpisodeRewardBufferNoBias):
            """Format examples showing L, U matrices and rewards."""
            if len(replay_buffer.buffer) == 0:
                return "(No previous attempts yet)\n"
            
            text = f"Total previous attempts: {len(replay_buffer.buffer)}\n"
            text += "=" * 60 + "\n\n"
            
            for idx, (weights, reward) in enumerate(replay_buffer.buffer, 1):
                # weights should be a dict with L, U, bias
                if isinstance(weights, dict) and 'L' in weights:
                    L = weights['L']
                    U = weights['U']
                    text += f"Attempt #{idx}:\n"
                    text += "L matrix:\n"
                    for row in L:
                        text += ", ".join([f"{x:.2g}" for x in row]) + "\n"
                    text += "U matrix:\n"
                    for row in U:
                        text += ", ".join([f"{x:.2g}" for x in row]) + "\n"
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
        print("Updating the policy...")
        
        if self.use_lu_factorization:
            # Use LU factorization approach
            new_lu_components, reasoning, api_time = self.llm_brain.llm_update_parameters_num_optim(
                str_lu_examples(self.replay_buffer),
                parse_lu_matrices,
                self.training_episodes,
                self.rank,
                self.optimum,
                self.search_step_size,
                dim_state=self.policy.dim_states,  # Use policy's original dim_states
                dim_action=self.policy.dim_actions,  # Use policy's original dim_actions
                lu_rank=self.lu_rank,
                use_lu=True
            )
            self.api_call_time += api_time
            
            print(f"L shape: {new_lu_components['L'].shape}, U shape: {new_lu_components['U'].shape}")
            self.policy.update_policy(lu_components=new_lu_components)
            print(f"Weight shape after update: {self.policy.weight.shape}")
            
            # Store LU components for replay buffer
            new_parameter_list = new_lu_components
        else:
            # Use regular parameter optimization
            new_parameter_list, reasoning, api_time = self.llm_brain.llm_update_parameters_num_optim(
                str_nd_examples(self.replay_buffer, self.rank),
                parse_parameters,
                self.training_episodes,
                self.rank,
                self.optimum,
                self.search_step_size
            )
            self.api_call_time += api_time

            print(self.policy.get_parameters().shape)
            print(new_parameter_list.shape)
            self.policy.update_policy(new_parameter_list)
            print(self.policy.get_parameters().shape)
        
        logging_q_filename = f"{logdir}/parameters.txt"
        logging_q_file = open(logging_q_filename, "w")
        logging_q_file.write(str(self.policy))
        logging_q_file.close()
        q_reasoning_filename = f"{logdir}/parameters_reasoning.txt"
        q_reasoning_file = open(q_reasoning_filename, "w")
        q_reasoning_file.write(reasoning)
        q_reasoning_file.close()
        print("Policy updated!")

        # Run the episode and collect the trajectory
        print(f"Rolling out episode {self.training_episodes}...")
        logging_filename = f"{logdir}/training_rollout.txt"
        logging_file = open(logging_filename, "w")
        results = []
        for idx in range(self.num_evaluation_episodes):
            if idx == 0:
                result = self.rollout_episode(world, logging_file, record=False)
            else:
                result = self.rollout_episode(world, logging_file, record=False)
            results.append(result)
        print(f"Results: {results}")
        result = np.mean(results)
        self.replay_buffer.add(new_parameter_list, result)
        
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

        # Clear GPU memory after episode to prevent OOM
        self.llm_brain.clear_cache()

        self.training_episodes += 1

        _cpu_time = time.process_time() - self.start_time
        _api_time = self.api_call_time
        _total_episodes = self.total_episodes
        _total_steps = self.total_steps
        _total_reward = result
        return _cpu_time, _api_time, _total_episodes, _total_steps, _total_reward
    
    def plot_reward_progress(self, logdir):
        """Plot training episode rewards (excluding warmup)."""
        if len(self.training_rewards) == 0:
            return  # No training rewards to plot yet
            
        plt.figure(figsize=(10, 6))
        episodes = list(range(len(self.training_rewards)))
        
        plt.plot(episodes, self.training_rewards, 'b-', marker='o', markersize=4, linewidth=2)
        
        plt.xlabel('Training Episode', fontsize=12)
        plt.ylabel('Reward', fontsize=12)
        plt.title('Training Reward Progress', fontsize=14)
        plt.grid(True, alpha=0.3)
        
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
        
        # If using LU factorization, also save L and U heatmaps (generated iteratively each episode)
        if self.use_lu_factorization and hasattr(self.policy, 'L') and self.policy.L is not None:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # L matrix
            sns.heatmap(
                self.policy.L,
                annot=True,
                fmt='.2f',
                cmap='viridis',
                ax=axes[0],
                cbar_kws={'label': 'Value'}
            )
            axes[0].set_title('L Matrix')
            axes[0].set_xlabel('LU Rank Dimension')
            axes[0].set_ylabel('State Dimension')
            
            # U matrix
            sns.heatmap(
                self.policy.U_matrix,
                annot=True,
                fmt='.2f',
                cmap='viridis',
                ax=axes[1],
                cbar_kws={'label': 'Value'}
            )
            axes[1].set_title('U Matrix')
            axes[1].set_xlabel('Action Dimension')
            axes[1].set_ylabel('LU Rank Dimension')
            
            # Reconstructed weight (L @ U)
            sns.heatmap(
                weight_matrix,
                annot=True,
                fmt='.2f',
                cmap='viridis',
                ax=axes[2],
                cbar_kws={'label': 'Value'}
            )
            axes[2].set_title('Weight (L @ U)')
            axes[2].set_xlabel('Action Dimension')
            axes[2].set_ylabel('State Dimension')
            
            plt.suptitle(f'LU Factorization - Episode {self.training_episodes}', fontsize=16)
            plt.tight_layout()
            
            lu_plot_filename = f"{logdir}/policy_lu_heatmap_ep{self.training_episodes}.png"
            plt.savefig(lu_plot_filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved LU factorization heatmap to {lu_plot_filename}")
    
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
        
        # Create LU factorization heatmap GIF if using LU factorization
        lu_gif = None
        if self.use_lu_factorization:
            print("\nCreating LU Factorization Heatmap GIF...")
            lu_gif = create_gif("policy_lu_heatmap", "policy_lu_heatmaps.gif")
        
        print("=" * 60)
        return policy_gif, lu_gif

    def evaluate_policy(self, world: BaseWorld, logdir):
        results = []
        for idx in range(self.num_evaluation_episodes):
            logging_filename = f"{logdir}/evaluation_rollout_{idx}.txt"
            logging_file = open(logging_filename, "w")
            result = self.rollout_episode(world, logging_file, record=False)
            results.append(result)
        return results
