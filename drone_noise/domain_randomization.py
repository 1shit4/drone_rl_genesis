import torch

def add_observation_noise(obs_buf, noise_level=0.01):
    """
    Adds Gaussian noise to the observation buffer to mimic sensor noise.
    """
    if noise_level > 0:
        noise = torch.randn_like(obs_buf) * noise_level
        return obs_buf + noise
    return obs_buf


def randomize_mass(drone, envs_idx, base_masses, randomization_params):
    """
    (REAL DR) Randomizes the mass of the drone's links for specified environments.
    This uses the simulator's API to change the underlying physics.
    """
    if 'mass_scale' not in randomization_params or len(envs_idx) == 0:
        return

    mass_range = randomization_params['mass_scale']
    num_links = drone.num_links
    num_envs_to_reset = len(envs_idx)
    
    # Generate random scales for each link in each resetting environment
    scales = (mass_range[1] - mass_range[0]) * torch.rand(
        (num_envs_to_reset, num_links), device=drone.device
    ) + mass_range[0]
    
    # Calculate and set the new masses
    new_masses = base_masses * scales
    drone.set_links_inertial_mass(new_masses, envs_idx=envs_idx)


def generate_mimicked_parameters(num_envs_to_reset, randomization_params, device):
    """
    (MIMICKED DR) Generates randomized parameters for mimicking physics changes.
    This function does NOT change the simulator, it only provides the values
    to be used in the controller logic (step function).
    """
    mimicked_params = {}

    # --- Mimic kf/km variations by generating actuator efficiency scales ---
    if 'actuator_scale' in randomization_params:
        actuator_range = randomization_params['actuator_scale']
        scales = (actuator_range[1] - actuator_range[0]) * torch.rand(
            (num_envs_to_reset,), device=device
        ) + actuator_range[0]
        mimicked_params['actuator_scales'] = scales

    # --- Mimic inertia variations by generating action smoothing factors ---
    if 'action_smoothing_scale' in randomization_params:
        smoothing_range = randomization_params['action_smoothing_scale']
        # Alpha of 1.0 = no smoothing. Lower alpha = more smoothing (heavier feel)
        alphas = (smoothing_range[1] - smoothing_range[0]) * torch.rand(
            (num_envs_to_reset, 3), device=device  # One alpha per body rate axis
        ) + smoothing_range[0]
        mimicked_params['smoothing_alphas'] = alphas

    return mimicked_params