import yaml
import argparse
from runner import llm_num_optim_runner, llm_num_optim_semantics_runner
# import gym_maze
# import gym_navigation
from envs import nim, pong


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the config file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if config["task"] in ["cont_space_llm_num_optim", "cont_space_llm_num_optim_rndm_proj", "dist_state_llm_num_optim"]:
        llm_num_optim_runner.run_training_loop(**config)
    elif config["task"] in ["dist_state_llm_num_optim_semantics", "cont_state_llm_num_optim_semantics"]:
        llm_num_optim_semantics_runner.run_training_loop(**config)
    else:
        raise ValueError(f"Task {config['task']} not recognized.")


if __name__ == "__main__":
    main()
