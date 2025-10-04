import numpy as np
from Sensors_Models.utils import evaluate_range_beam_dist_array as sensor_model
from Sensors_Models.ray_casting import cast_rays

def hill_climb_binary(objective, n_bits=20, n_iterations=1000, objective_args=()):
    """
    Hill Climbing for binary optimization problems.

    Args:
        objective: function to minimize, takes a binary vector
        n_bits: length of binary vector (solution)
        n_iterations: number of iterations

    Returns:
        best_solution, best_eval
    """
    # Initial random binary solution
    solution = np.random.randint(0, 2, n_bits)
    # print("Initial solution:", solution)
    solution_eval = objective(solution, *objective_args)

    best, best_eval = solution.copy(), solution_eval
    early_count = 0

    for it in range(n_iterations):
        if (it+1) % (n_iterations//100) == 0 or it == 0:
            print(f"Iteration {it+1}/{n_iterations}, current best eval: {best_eval}")
        # Generate neighbor by flipping one random bit
        candidate = solution.copy()
        flip_index = np.random.randint(0, n_bits, size=5)
        # print("Flipping index:", flip_index)
        candidate[flip_index] = 1 - candidate[flip_index]  # flip 0->1 or 1->0
        candidate_eval = objective(candidate, *objective_args)

        # Accept if better
        if candidate_eval > solution_eval:
            solution, solution_eval = candidate, candidate_eval
            if candidate_eval > best_eval:
                best, best_eval = candidate.copy(), candidate_eval
            early_count = 0
        else:
            early_count += 1
        
        # Early stopping if optimal solution found or best_eval 
        if best_eval >= 0 or early_count >= 1000:
            print("Early stopping criteria met.")
            break

    return best, best_eval

def objective_MAP_occupancy_grid_mapping_hill_climb(occ_map_flat, robot_poses, ranges, z_max, num_rays, fov):
    """
    Objective function for MAP occupancy grid mapping to be maximized.

    Args:
        occ_map_flat: flattened binary occupancy map (0-free, 1-occupied)
        robot_poses: list of robot poses [x, y, theta]
        ranges: list of laser ranges
        z_max: maximum range of the laser
        num_rays: number of rays in the laser scan
        fov: field of view of the laser

    Returns:
        log-likelihood of the map given the measurements
    """

    # Reshape flat map to 2D grid
    grid_size = int(np.sqrt(len(occ_map_flat)))
    occ_map = occ_map_flat.reshape((grid_size, grid_size))

    l0 = 0.5  # prior log-odds value
    mix_density, sigma, lamb_short = [0.7, 0.2, 0.05, 0.05], 0.75, 0.9
    total_log_likelihood = 0.0

    for pose, z in zip(robot_poses, ranges):
        # Get end points of rays based on current pose and measurement
        # end_points = find_endpoints(pose, z, num_rays, fov)
        _, z_star = cast_rays(occ_map, pose, num_rays, fov, z_max)

        # evaluate the measurement model p(z|x,m)
        p_hit, p_short, p_max, p_rand, p_z = sensor_model(z, z_star, z_max, mix_density, sigma, lamb_short)
        # print(f"t={t}, z={z}, z*={z_star}, p_hit={p_hit}, p_short={p_short}, p_max={p_max}, p_rand={p_rand}, p_z={p_z}")
        total_log_likelihood += np.log(p_z + 1e-6)
    # Add prior log-odds
    total_log_likelihood = np.sum(total_log_likelihood) + np.sum(occ_map_flat * l0)

    return total_log_likelihood
