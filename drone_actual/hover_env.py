import torch
import math
import copy
import pathlib
import genesis as gs
from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


script_path = pathlib.Path(__file__).resolve()
project_root = script_path.parent.parent
urdf_file_name = "Tarot 650 Assembly_urdf_wts.SLDASM.urdf"
urdf_path = project_root / "custom urdf" / "Tarot 650 Assembly_urdf_wts.SLDASM" / "urdf" / urdf_file_name

class HoverEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.rendered_env_num = min(10, self.num_envs)
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.dt = 0.01  # run in 100hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        #pid vars
        self.error_prev = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.integral = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = copy.deepcopy(reward_cfg["reward_scales"])

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=env_cfg["max_visualize_FPS"],
                camera_pos=(3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(self.rendered_env_num))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plane
        self.scene.add_entity(gs.morphs.Plane())

        # add target
        if self.env_cfg["visualize_target"]:
            self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=0.05,
                    fixed=False,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=(1.0, 0.5, 0.5),
                    ),
                ),
            )
        else:
            self.target = None

        # add camera
        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=True,
            )

        # add drone
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.drone = self.scene.add_entity(gs.morphs.Drone(file=urdf_path))

        # build scene
        self.scene.build(n_envs=num_envs)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # initialize buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.last_base_pos = torch.zeros_like(self.base_pos)

        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["pos_x_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["pos_y_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["pos_z_range"], (len(envs_idx),), gs.device)

    def _at_target(self):
        return (
            (torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"])
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.actions

        # 14468 is hover rpm
        # self.drone.set_propellels_rpm((1 + exec_actions * 0.8) * 14468.429183500699)
        mass = 2.267
        g = 9.81
        
        inv_mixer = torch.tensor([
          [ 0.25, -1.5714,  1.5714, -22.3694],
          [ 0.25,  1.5714,  1.5714,  22.3694],
          [ 0.25,  1.5714, -1.5714, -22.3694],
          [ 0.25, -1.5714, -1.5714,  22.3694]
        ], device=gs.device)
        #kf = 3.16e-10  * 49 
        kf = 1.02e-8

        max_thrust = mass*g*2.25##max thrust is 0.5 x weight
        max_ang_vel = 2*math.pi #max angular velocity is 2* pi rad/s
        ang_vel = self.base_ang_vel
        thrust = ((exec_actions[:,0] + 1)/2)*max_thrust

        Moment = self.pid(exec_actions[:,1:4]*max_ang_vel, ang_vel, self.error_prev, self.integral)
       
        # --- START CORRECT DIAGNOSTIC BLOCK ---
        # This checks for invalid values right after they are calculated and before they crash the simulation.
        if torch.isnan(thrust).any() or torch.isinf(thrust).any() or torch.isnan(Moment).any() or torch.isinf(Moment).any():
            print("\n" + "="*80)
            print("!!! DIAGNOSTIC: Invalid values DETECTED in Thrust/Moment calculation!")
            print(f"    Thrust (has_nan / has_inf): {torch.isnan(thrust).any()} / {torch.isinf(thrust).any()}")
            print(f"    Moment (has_nan / has_inf): {torch.isnan(Moment).any()} / {torch.isinf(Moment).any()}")
            print(f"    Integral (min/max/has_inf): {torch.min(self.integral):.2f} / {torch.max(self.integral):.2f} / {torch.isinf(self.integral).any()}")
            print("="*80 + "\n")
        # --- END CORRECT DIAGNOSTIC BLOCK ---

        cin = torch.stack([thrust, Moment[:,0], Moment[:,1], Moment[:,2]], dim=1)
        m_t = torch.matmul(inv_mixer, cin.T).T

        rpm = torch.sqrt(torch.clamp(m_t, min=0)/kf) * (60 / (2*math.pi))/50
  
        self.drone.set_propellels_rpm(rpm)

        # update target pos
        if self.target is not None:
            self.target.set_pos(self.commands, zero_velocity=True)
        self.scene.step()

        # --- START POST-PHYSICS DIAGNOSTIC ---
        # Check the state of ALL drones for corruption immediately after the physics engine runs.
        all_pos = self.drone.get_pos()
        all_quat = self.drone.get_quat()
        all_vel = self.drone.get_vel()
        all_ang = self.drone.get_ang()

        if torch.isnan(all_pos).any() or torch.isinf(all_pos).any() or torch.isnan(all_vel).any() or torch.isinf(all_vel).any():
            print("\n" + "="*80)
            print("!!! DIAGNOSTIC: STATE CORRUPTION DETECTED AFTER PHYSICS STEP !!!")
            print(f"    NaN/Inf in positions: {torch.isnan(all_pos).any() or torch.isinf(all_pos).any()}")
            print(f"    NaN/Inf in rotations: {torch.isnan(all_quat).any() or torch.isinf(all_quat).any()}")
            print(f"    NaN/Inf in linear velocities: {torch.isnan(all_vel).any() or torch.isinf(all_vel).any()}")
            print(f"    NaN/Inf in angular velocities: {torch.isnan(all_ang).any() or torch.isinf(all_ang).any()}")
            print("="*80 + "\n")
        # --- END POST-PHYSICS DIAGNOSTIC ---


        # update buffers
        self.episode_length_buf += 1
        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:] = self.drone.get_pos()
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[:] = self.drone.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(
                torch.ones_like(self.base_quat) * self.inv_base_init_quat,
                self.base_quat,
            ),
            rpy=True,
            degrees=True,
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.drone.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.drone.get_ang(), inv_base_quat)

        # resample commands
        envs_idx = self._at_target()
        self._resample_commands(envs_idx)

        # check termination and reset
        self.crash_condition = (
            (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
            | (torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
            | (torch.abs(self.rel_pos[:, 0]) > self.env_cfg["termination_if_x_greater_than"])
            | (torch.abs(self.rel_pos[:, 1]) > self.env_cfg["termination_if_y_greater_than"])
            | (torch.abs(self.rel_pos[:, 2]) > self.env_cfg["termination_if_z_greater_than"])
            | (self.base_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])
        )
        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | self.crash_condition

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_buf = torch.cat(
            [
                torch.clip(self.rel_pos * self.obs_scales["rel_pos"], -1, 1),
                self.base_quat,
                torch.clip(self.base_lin_vel * self.obs_scales["lin_vel"], -1, 1),
                torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1, 1),
                self.last_actions,
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def pid(self, target, current, error_prev, integral):
    
        kp = torch.tensor([0.00005, 0.00005, 0.00003], device=gs.device)
        kd = torch.tensor([0.0000005, 0.0000005, 0.0000003], device=gs.device)
        ki = torch.tensor([0.0000005, 0.0000005, 0.0000003], device=gs.device)
        
        error = target - current
        derivative = (error - error_prev) / self.dt
        integral = integral + error * self.dt

        #clamping integral to prevent windup
        integral = torch.clamp(integral, -10.0, 10.0)


        output = kp * error + kd * derivative + ki *integral
        
        self.error_prev[:] = error  # update previous error
        self.integral[:] = integral  # update integral term
        return output

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # --- START CORRECT DIAGNOSTIC BLOCK ---
        # Check for NaNs in the state tensors of the environments that are about to be reset.
        if torch.isnan(self.base_pos[envs_idx]).any() or \
           torch.isnan(self.base_quat[envs_idx]).any() or \
           torch.isnan(self.base_lin_vel[envs_idx]).any() or \
           torch.isnan(self.base_ang_vel[envs_idx]).any():
            print("\n" + "="*80)
            print("!!! DIAGNOSTIC: NaNs DETECTED in drone state BEFORE reset!")
            print(f"    Affected env_ids: {envs_idx.tolist()}")
            print(f"    NaNs in positions: {torch.isnan(self.base_pos[envs_idx]).any()}")
            print(f"    NaNs in rotations: {torch.isnan(self.base_quat[envs_idx]).any()}")
            print(f"    NaNs in linear velocities: {torch.isnan(self.base_lin_vel[envs_idx]).any()}")
            print(f"    NaNs in angular velocities: {torch.isnan(self.base_ang_vel[envs_idx]).any()}")
            print("="*80 + "\n")
        # --- END CORRECT DIAGNOSTIC BLOCK ---
        
        
        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.last_base_pos[envs_idx] = self.base_init_pos
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.drone.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.drone.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        #reset pid
        self.error_prev[envs_idx] = 0.0
        self.integral[envs_idx] = 0.0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_target(self):
        target_rew = torch.sum(torch.square(self.last_rel_pos), dim=1) - torch.sum(torch.square(self.rel_pos), dim=1)
        return target_rew

    def _reward_smooth(self):
        smooth_rew = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        return smooth_rew

    def _reward_yaw(self):
        yaw = self.base_euler[:, 2]
        yaw = torch.where(yaw > 180, yaw - 360, yaw) / 180 * 3.14159  # use rad for yaw_reward
        yaw_rew = torch.exp(self.reward_cfg["yaw_lambda"] * torch.abs(yaw))
        return yaw_rew

    def _reward_angular(self):
        angular_rew = torch.norm(self.base_ang_vel / 3.14159, dim=1)
        return angular_rew

    def _reward_crash(self):
        crash_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1
        return crash_rew
    
    def _reward_stay_on_target(self):
        distance_to_target = torch.norm(self.rel_pos, dim=1)
        #stay_rew = torch.where(distance_to_target < self.env_cfg["at_target_threshold"], 1.0, 0.0)
        stay_rew = torch.exp(self.reward_cfg["target_lambda"] * distance_to_target)
        return stay_rew
