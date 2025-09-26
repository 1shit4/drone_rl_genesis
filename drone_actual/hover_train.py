import faulthandler
faulthandler.enable()
import argparse
import os
import pickle
import shutil
from importlib import metadata

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


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.004,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 6,
            "num_mini_batches": 64,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "tanh",
            "actor_hidden_dims": [256, 128],
            "critic_hidden_dims": [256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 400,
        "save_interval": 100,
        "empirical_normalization": True,
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 4,
        # termination
        "termination_if_roll_greater_than": 90,  # degree
        "termination_if_pitch_greater_than": 90,
        "termination_if_close_to_ground": 0.3,
        "termination_if_x_greater_than": 5.0,
        "termination_if_y_greater_than": 5.0,
        "termination_if_z_greater_than": 2.0,
        # base pose
        "base_init_pos": [0.0, 0.0, 2.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 60.0,
        "at_target_threshold": 0.05,
        "resampling_time_s": 3.0,
        "simulate_action_latency": True,
        "clip_actions": 1.0,
        # visualization
        "visualize_target": False,
        "visualize_camera": False,
        "max_visualize_FPS": 60,
    }
    obs_cfg = {
        "num_obs": 17,
        "obs_scales": {
            "rel_pos": 1 / 3.0,
            "lin_vel": 1 / 8.0,
            "ang_vel": 1 / 3.14159,
        },
    }
    reward_cfg = {
        "yaw_lambda": -10.0,
        "target_lambda": -0.8,
        "reward_scales": {
            "target": 10.0,
         #  "smooth": -1e-4,
         #   "yaw": 0.01,
         #   "angular": -2e-4,
            "crash": -10.0,
            "stay_on_target": 2.0,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "pos_x_range": [1, 4],
        "pos_y_range": [1, 4],
        "pos_z_range": [1.0, 3.0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="drone-hovering")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=8192)
    parser.add_argument("--max_iterations", type=int, default=301)
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training from a checkpoint")
    parser.add_argument("--load_model", type=str, default=None, help="Path to the model checkpoint to load")
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if not args.resume and os.path.exists(log_dir):
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)

    if args.vis:
        env_cfg["visualize_target"] = True

    if not args.resume:
        pickle.dump(
            [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
            open(f"{log_dir}/cfgs.pkl", "wb"),
        )
    if args.resume:
        env_cfg, obs_cfg, reward_cfg, command_cfg, _ = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
        train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    env = HoverEnv(
            num_envs=args.num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=args.vis,
        )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    if args.resume:
        resume_path = args.load_model
        print(f"Resuming training from checkpoint: {resume_path}")
        runner.load(resume_path)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python examples/drone/hover_train.py
"""
