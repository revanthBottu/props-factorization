from world.continuous_space_general_world import ContinualSpaceGeneralWorld
from world.discrete_state_general_world import DiscreteStateGeneralWorld
from agent.llm_num_optim_linear_policy_rndm_proj import LLMNumOptimRndmPrjAgent
from agent.llm_num_optim_linear_policy import LLMNumOptimAgent
from agent.llm_num_optim_q_table import LLMNumOptimQTableAgent
from jinja2 import Environment, FileSystemLoader
import os
import traceback
import numpy as np


def run_training_loop(
    task,
    num_episodes,
    gym_env_name,
    render_mode,
    logdir,
    dim_actions,
    dim_states,
    max_traj_count,
    max_traj_length,
    template_dir,
    llm_si_template_name,
    llm_output_conversion_template_name,
    llm_model_name,
    num_evaluation_episodes,
    warmup_episodes,
    warmup_dir,
    bias=None,
    rank=None,
    optimum=1000,
    search_step_size=0.1,
    env_kwargs=None,
    use_factorized_policy=False,
    factor_rank=None,
):
    assert task in ["cont_space_llm_num_optim", "cont_space_llm_num_optim_rndm_proj", "dist_state_llm_num_optim"]

    jinja2_env = Environment(loader=FileSystemLoader(template_dir))
    llm_si_template = jinja2_env.get_template(llm_si_template_name)
    llm_output_conversion_template = jinja2_env.get_template(
        llm_output_conversion_template_name
    )

    if task in ["cont_space_llm_num_optim", "cont_space_llm_num_optim_rndm_proj"]:
        world = ContinualSpaceGeneralWorld(
            gym_env_name,
            render_mode,
            max_traj_length,
        )


        if task == "cont_space_llm_num_optim":
            agent = LLMNumOptimAgent(
                logdir,
                dim_actions,
                dim_states,
                max_traj_count,
                max_traj_length,
                llm_si_template,
                llm_output_conversion_template,
                llm_model_name,
                num_evaluation_episodes,
                bias,
                optimum,
                search_step_size,
                use_factorized_policy=use_factorized_policy,
                factor_rank=factor_rank,
            )
        elif task == "cont_space_llm_num_optim_rndm_proj":
            agent = LLMNumOptimRndmPrjAgent(
                logdir,
                dim_actions,
                dim_states,
                max_traj_count,
                max_traj_length,
                llm_si_template,
                llm_output_conversion_template,
                llm_model_name,
                num_evaluation_episodes,
                rank,
                bias,
                optimum,
            )


    elif task == "dist_state_llm_num_optim":
        world = DiscreteStateGeneralWorld(
            gym_env_name,
            render_mode,
            max_traj_length,
            env_kwargs=env_kwargs,
        )

        agent = LLMNumOptimQTableAgent(
            logdir,
            dim_actions,
            dim_states,
            max_traj_count,
            max_traj_length,
            llm_si_template,
            llm_output_conversion_template,
            llm_model_name,
            num_evaluation_episodes,
            optimum,
            env_kwargs=env_kwargs,
        )

        print('init done')

    if not warmup_dir:
        warmup_dir = f"{logdir}/warmup"
        os.makedirs(warmup_dir, exist_ok=True)
        agent.random_warmup(world, warmup_dir, warmup_episodes)
    else:
        agent.replay_buffer.load(warmup_dir)
    
    overall_log_file = open(f"{logdir}/overall_log.txt", "w")
    overall_log_file.write("Iteration, CPU Time, API Time, Total Episodes, Total Steps, Total Reward, Variance, Std\n")
    overall_log_file.flush()
    for episode in range(num_episodes):
        print(f"Episode: {episode}")
        # create log dir
        curr_episode_dir = f"{logdir}/episode_{episode}"
        print(f"Creating log directory: {curr_episode_dir}")
        os.makedirs(curr_episode_dir, exist_ok=True)
        
        for trial_idx in range(5):
            try:
                cpu_time, api_time, total_episodes, total_steps, total_reward, variance, std = agent.train_policy(world, curr_episode_dir)
                overall_log_file.write(f"{episode + 1}, {cpu_time}, {api_time}, {total_episodes}, {total_steps}, {total_reward}, {variance}, {std}\n")
                overall_log_file.flush()
                print(f"{trial_idx + 1}th trial attempt succeeded in training")
                break
            except Exception as e:
                print(
                    f"{trial_idx + 1}th trial attempt failed with error in training: {e}"
                )
                traceback.print_exc()

                if trial_idx == 4:
                    print(f"All {trial_idx + 1} trials failed. Train terminated")
                    exit(1)
                continue
    overall_log_file.close()
    
    # Generate animated GIFs from heatmaps after training completes
    if task in ["cont_space_llm_num_optim", "cont_space_llm_num_optim_rndm_proj"]:
        print("\n" + "=" * 70)
        print("Training Complete! Generating animated GIFs from heatmaps...")
        print("=" * 70)
        try:
            agent.create_heatmap_gifs(logdir, duration=500, loop=0)
        except Exception as e:
            print(f"Warning: Could not create GIFs: {e}")
            traceback.print_exc()
