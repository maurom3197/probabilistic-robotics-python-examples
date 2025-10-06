import math
import numpy as np

from Mapping.gridmap_utils import compute_map_occ, plot_gridmap, get_map

from Discrete_Filters.utils import (
    residual, 
    initial_uniform_particles_gridmap, initial_uniform_particles_gridmap_from_free_spaces, initial_gaussian_particles,
    state_mean, 
    simple_resample, stratified_resample, systematic_resample, residual_resample,
)

from Discrete_Filters.plot_utils import plot_initial_particles_gridmap, plot_particles_gridmap
import matplotlib.pyplot as plt
from Discrete_Filters.pf import RobotPF

from Discrete_Filters.probabilistic_models import (
    sample_velocity_motion_model,
    sample_odometry_motion_model,
    get_odometry_command,
    landmark_range_bearing_model,
    landmark_range_bearing_sensor,
    likelihood_field_laser_model,
)

from Sensors_Models.likelihood_fields import compute_distances, precompute_likelihood_field, plot_likelihood_fields
from Sensors_Models.ray_casting import cast_rays, plot_rays_on_gridmap
from Sensors_Models.utils import compute_p_hit_dist, evaluate_p_hit

def run_localization_sim(
    pf: RobotPF,
    pf_dt,
    landmarks,
    map,
    landm_max_range,
    landm_fov,
    lidar_max_range,
    lidar_fov,
    lidar_num_rays,
    z_landm_sensor,
    eval_hx_landm,
    eval_hx_lidar,
    sigma_u,
    sigma_z_landm,
    mix_density_lidar,
    distances,
    sigma_z_lidar,
    motion_model="velocity",
    sigma_u_odom=0.0,
    sim_step_s=0.1,
    particles_plot_step_s=5.0,
    sim_length_s=1,
):

    sim_pos = pf.mu.copy()  # simulated position, copy the initial position set inside the PF
    odom_pos = pf.mu.copy()  # odometry position, copy the initial position set inside the PF

    cmd_vel = np.array(
        [0.8, 0.05]
    )  # velocity command (v, omega). In this case will be constant for the whole simulation

    # convert the durations to number of time steps
    steps = int(sim_length_s / sim_step_s)
    pf_step = int(pf_dt / sim_step_s)
    particles_plot_step = int(particles_plot_step_s / sim_step_s)

    # Initialize a plot and insert the landmarks
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    plot_gridmap(map, ax=ax[0])
    # fig_particles, ax_particles = plt.subplots(1, 1, figsize=(6, 6))
    #lmarks_legend = ax[0].scatter(landmarks[:, 1], map.shape[0]-landmarks[:, 0], marker="s", s=60, label="Landmarks")

    track = []  # list to store all the robot positions
    track_odom = []  # list to store all the odometry positions
    track_pf = [pf.mu.copy()]  # list to store all the pf positions

    odom_pos_prev = odom_pos.copy()

    # plot initial distribution of particles
    initial_particles_legend = plot_initial_particles_gridmap(pf.N, pf.particles, map_shape=map.shape, ax=ax[0])

    # The main loop that runs the simulation
    for i in range(steps):
        if np.any(cmd_vel == 0.0):
            cmd_vel += 1e-9
        elif i > steps/2:
            cmd_vel = np.array([0.65, -0.03])
        # Simulate robot motion for sim_step_s seconds using the Motion Model.
        # the sampling motion model already include Gaussian noise on the command
        sim_pos = sample_velocity_motion_model(sim_pos, cmd_vel, sigma_u, sim_step_s)
        track.append(sim_pos)

        # to simulate the error in the odometry reading, we take another Gaussian sample of the velocity command
        odom_pos = sample_velocity_motion_model(odom_pos, cmd_vel, sigma_u, sim_step_s)
        track_odom.append(odom_pos)

        if i % pf_step == 0 and i != 0:  # only update pf at dt intervals
            # run the prediction step of the PF
            if motion_model == "velocity":
                pf.predict(u=cmd_vel, sigma_u=sigma_u, g_extra_args=(pf_dt,))
            elif motion_model == "odometry":
                u = get_odometry_command(odom_pos, odom_pos_prev)
                pf.predict(u=u, sigma_u=sigma_u_odom)
                odom_pos_prev = odom_pos.copy()

            pf.estimate(mean_fn=state_mean, residual_fn=residual, angle_idx=2)

            # for each landmark simulate the measurement of the landmark
            # for lmark in landmarks:
            #     z = z_landm_sensor(
            #         sim_pos, lmark, sigma_z_landm, max_range=landm_max_range, fov=landm_fov
            #     )  # landmarks out of the sensor's FOV will be not detected

            #     # if any landmark detected by the sensor, update the PF
            #     if z is not None:
            #         # run the correction step of the PF
            #         pf.update(z, sigma_z_landm, eval_hx=eval_hx_landm, hx_args=(lmark, sigma_z_landm))
            
            # simulate laser measurement adding noise to the ones obtained by casting rays in the map
            end_points, rng = cast_rays(map, pf.mu, lidar_num_rays, lidar_fov, lidar_max_range) # with real laser sensor this is not needed
            z_points = rng + np.random.normal(0, 0.1**2, size=1).item() + np.random.binomial(2, 0.001, 1).item() + 10*np.random.binomial(2, 0.001, 1).item()
            z_points = np.clip(z_points, 0., lidar_max_range)

            # update filter state with likelihood field model
            pf.update(z_points, sigma_z_lidar, eval_hx=eval_hx_lidar, 
                      hx_args=(z_points, distances, sigma_z_lidar, lidar_num_rays, lidar_max_range, lidar_fov, mix_density_lidar), 
                      z_prob=True
                      )

            # after the update of the weights with the measurements, we normalize the weights to make them probabilities
            pf.normalize_weights()

            # resample if too few effective particles
            neff = pf.neff()
            print(np.max(pf.weights))
            if neff < 2*pf.N / 3:
                print(np.max(pf.weights))
                pf.resampling(
                    resampling_fn=pf.resampling_fn,  # simple, residual, stratified, systematic
                    resampling_args=(pf.weights,),  # tuple: only pf.weights if using pre-defined functions
                )
                assert np.allclose(pf.weights, 1 / pf.N)

            # estimate robot mean and covariance from particles
            pf.estimate(mean_fn=state_mean, residual_fn=residual, angle_idx=2)

            # plot the posterior particles every particles_plot_step seconds
            if i % particles_plot_step == 0:
                legend_PF1, legend_PF2 = plot_particles_gridmap(pf.particles, sim_pos, pf.mu, map.shape, ax=ax[0])
            track_pf.append(pf.mu.copy())

            print("Step: ", i, " - NEFF: ", neff)

    # draw plots
    track = np.array(track)
    track_odom = np.array(track_odom)
    track_pf = np.array(track_pf)

    # trajectory plots
    (track_legend,) = ax[0].plot(track[:, 1], map.shape[0]*np.ones_like(track[:,0])-track[:, 0], label="Real robot path")
    (track_odom_legend,) = ax[0].plot(track_odom[:, 1], map.shape[0]*np.ones_like(track_odom[:,0])-track_odom[:, 0], "--", label="Odometry path")
    ax[0].axis("equal")
    ax[0].set_title("PF Robot localization Gridmap")
    ax[0].legend(handles=[track_legend, track_odom_legend, legend_PF1, legend_PF2, initial_particles_legend])

    # error plots
    (pf_err,) = ax[1].plot(
        np.arange(0, sim_length_s, pf_dt),
        np.linalg.norm(track[::pf_step, :2] - track_pf[:, :2], axis=1),
        "-o",
        label="PF error",
    )
    (odom_err,) = ax[1].plot(
        np.arange(0, sim_length_s, sim_step_s),
        np.linalg.norm(track[:, :2] - track_odom[:, :2], axis=1),
        label="Odometry error",
    )
    ax[1].legend(handles=[pf_err, odom_err])
    ax[1].set_title("Robot path error")

    fig.suptitle("PF Robot localization - " + motion_model + "motion model")
    fig.tight_layout()

    plt.show()



def main():

    ##### Define Parameters #####

    seed = 42  # because it is the answer to the Ultimate Question of Life, The Universe and Everything :)
    np.random.seed(seed)

    # landmarks list in map's coordinate
    landmarks = np.array(
        #[[5, 12], [10.5, 7.5], [16.5, 15], [10, 14], [5, 6], [14.5, 11.5], [14, 9], [8, 15.5], [13.5, 17], [18.4, 18]]
        [[10.5, 12.5], [10.5,13.5], [19.5, 7.5], [19.5,8.5], [19.5,9.5], [18.5,9.5], [10.5,16.5], [11.5,16.5], [12.5,16.5], [16.5,16.5],
          [17.5, 16.5], [9.5,7.5], [11.5,4.5], [15.5,2.5], [0.5,12.5], [0.5,16.5]]
    )
    # sensor params
    landm_max_range = 8.0
    landm_fov = math.pi / 2

    # sim params
    pf_dt = 1.0  # time interval between measurements [s]
    sim_length_s = 28  # length of the simulation [s]

    # Probabilistic models parameters
    dim_x = 3
    # First, choose the Motion Model
    motion_model = "velocity"  # 'odometry' or 'velocity'

    # general noise parameters
    std_lin_vel = 0.15  # [m/s]
    std_ang_vel = np.deg2rad(2.0)  # [rad/s]
    sigma_u = np.array([std_lin_vel, std_ang_vel])
    sigma_u_odom = 0

    # Velocity motion model params
    if motion_model == "velocity":
        dim_u = 2
        eval_gux = sample_velocity_motion_model

    # odometry motion model params
    elif motion_model == "odometry":
        dim_u = 3
        std_rot1 = np.deg2rad(1.0)
        std_transl = 0.05
        std_rot2 = np.deg2rad(0.05)
        sigma_u_odom = np.array([std_rot1, std_transl, std_rot2])
        eval_gux = sample_odometry_motion_model

    # Define noise params and Q for landmark sensor model
    std_range = 0.1  # [m]
    std_bearing = np.deg2rad(1.0)  # [rad]
    sigma_z_landm = np.array([std_range, std_bearing])

    # Define gridmap
    map_path = '2D_maps/map3.png'

    xy_reso = 3
    _, grid_map = get_map(map_path, xy_reso)
    
    # print(grid_map)
    max_x = grid_map.shape[0]
    max_y = grid_map.shape[1]
    occ_spaces, free_spaces, map_spaces = compute_map_occ(grid_map)

    # Lidar sensor parameters
    lidar_max_range = 10.0
    lidar_num_rays = 60
    lidar_fov = 2*math.pi
    mix_density, sigma_z_lidar = [0.85, 0.05, 0.10], 0.75

    # To efficiently use Likelihood Field, pre-compute distances from obstacles in the map (from each map cell)
    distances = compute_distances(map_spaces, occ_spaces)

    # Plot gaussian likelihood fields
    # max_dist = 2.0 # this should be checked...
    # p_gridmap = precompute_likelihood_field(grid_map, sigma_z_lidar)
    # plot_likelihood_fields(p_gridmap, np.array([19, 2, 2*np.pi/3]))
    # plt.show()

    # reshape precomputed distances to have a lookup table with map size
    distances = np.reshape(distances, grid_map.shape)

    # Initialize the PF
    pf = RobotPF(
        dim_x=dim_x,
        dim_u=dim_u,
        eval_gux=eval_gux,
        resampling_fn=systematic_resample,
        boundaries=[(0.0, max_x), (0.0, max_y), (-np.pi, np.pi)],
        N=500,
    )

    pf.mu = np.array([19, 2, 2*np.pi/3])  # initial x, y, theta of the robot
    pf.Sigma = np.diag([0.1, 0.1, 0.1])   # initial covariance matrix
    
    # cast rays: compute end points and laser measurements
    # end_points, z_star = cast_rays(grid_map, pf.mu, lidar_num_rays, lidar_fov, lidar_max_range)
    # print("Perceived obstacles end points:", end_points)
    # print("Laser measurements:", z_star)
    # fig, ax = plt.subplots(figsize=(8,8))
    # pc = plot_rays_on_gridmap(grid_map, robot_pose=pf.mu, end_points=end_points, ax=ax)
    # fig.suptitle('Ray Casted on Grid Map', fontsize = 16)
    # plt.show()

    # initialize particles using the gridmap using the free spaces list
    init_particles_dist = "uniform_rejection"  # "uniform_free_spaces", "uniform_rejection", "gaussian"
    # method 1: use map pre-computed index list of free spaces
    if init_particles_dist == "uniform_free_spaces":
        pf.initialize_particles(
            initial_dist_fn=initial_uniform_particles_gridmap_from_free_spaces, 
            initial_dist_args=(pf.dim_x, free_spaces))
    # method 2: use rejection sampling
    elif init_particles_dist == "uniform_rejection":
        pf.initialize_particles(
            initial_dist_fn=initial_uniform_particles_gridmap, 
            initial_dist_args=(pf.dim_x, pf.boundaries, grid_map))
    # method 3: use a gaussian with apriori information on the initial robot pose 
    elif init_particles_dist == "gaussian":
        pf.initialize_particles(
            initial_dist_fn=initial_gaussian_particles, 
            initial_dist_args=(pf.dim_x, pf.mu, [3.0, 3.0, math.pi/2], 2, grid_map))

    run_localization_sim(
        pf,
        pf_dt=pf_dt,
        landmarks=landmarks,
        map=grid_map,
        z_landm_sensor=landmark_range_bearing_sensor,
        landm_max_range=landm_max_range,
        landm_fov=landm_fov,
        lidar_max_range=lidar_max_range,
        lidar_fov=lidar_fov,
        lidar_num_rays=lidar_num_rays,
        eval_hx_landm=landmark_range_bearing_model,
        eval_hx_lidar=likelihood_field_laser_model,
        sigma_u=sigma_u,
        sigma_z_landm=sigma_z_landm,
        sigma_z_lidar=sigma_z_lidar,
        mix_density_lidar=mix_density,
        distances=distances,
        motion_model=motion_model,
        sigma_u_odom=sigma_u_odom,
        particles_plot_step_s=3.0,
        sim_length_s=sim_length_s,
    )

    plt.close("all")


if __name__ == "__main__":
    main()
