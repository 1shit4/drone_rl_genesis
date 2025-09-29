import argparse
import os
import pickle
from importlib import metadata

import torch

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from hover_env import HoverEnv


def main():
    parser = argparse.ArgumentParser()
    # MODIFIED: Arguments now take the full path to the model and config files
    parser.add_argument("--model_path", type=str, required=True, help="Full path to the policy model (.pt file)")
    parser.add_argument("--config_path", type=str, required=True, help="Full path to the configuration file (cfgs.pkl)")
    parser.add_argument("--record", action="store_true", default=False, help="Enable video recording of the evaluation")
    args = parser.parse_args()

    gs.init()

    # MODIFIED: Load configurations from the user-specified path
    print(f"Loading configuration from: {args.config_path}")
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(args.config_path, "rb"))
    reward_cfg["reward_scales"] = {}

    # --- Environment and Visualization Settings ---
    env_cfg["visualize_target"] = True
    env_cfg["visualize_camera"] = args.record
    env_cfg["max_visualize_FPS"] = 60

    env = HoverEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    # Use the directory of the config file as the log_dir for the runner
    log_dir = os.path.dirname(args.config_path)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    # MODIFIED: Load the model from the user-specified path
    print(f"Loading model from: {args.model_path}")
    runner.load(args.model_path)
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()

    max_sim_step = int(env_cfg["episode_length_s"] * env_cfg["max_visualize_FPS"])
    with torch.no_grad():
        if args.record:
            print("Recording video...")
            env.cam.start_recording()
            for _ in range(max_sim_step):
                actions = policy(obs)
                obs, rews, dones, infos = env.step(actions)
                env.cam.render()
            video_filename = "hover_evaluation.mp4"
            env.cam.stop_recording(save_to_filename=video_filename, fps=env_cfg["max_visualize_FPS"])
            print(f"Video saved to: {video_filename}")
        else:
            print("Running simulation...")
            for _ in range(max_sim_step):
                actions = policy(obs)
                obs, rews, dones, infos = env.step(actions)
            print("Simulation finished.")


if __name__ == "__main__":
    main()