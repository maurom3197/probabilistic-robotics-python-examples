import math
import numpy as np

from Discrete_Filters.utils import residual, state_mean, simple_resample, stratified_resample, systematic_resample, residual_resample
from Discrete_Filters.plot_utils import plot_initial_particles, plot_particles
import matplotlib.pyplot as plt
from Discrete_Filters.pf import RobotPF

from Discrete_Filters.probabilistic_models import (
    sample_velocity_motion_model,
    sample_odometry_motion_model,
    get_odometry_command,
    landmark_range_bearing_model,
    landmark_range_bearing_sensor,
)


def run_localization_sim(
    pf: RobotPF,
    pf_dt,
    landmarks,
    max_range,
    fov,
    z_landm_sensor,
    eval_hx_landm,
    sigma_u,
    sigma_z,
    motion_model="velocity",
    sigma_u_odom=0.0,
    sim_step_s=0.1,
    particles_plot_step_s=5.0,
    sim_length_s=1,
):

    sim_pos = pf.mu.copy()  # simulated position, copy the initial position set inside the PF
    odom_pos = pf.mu.copy()  # odometry position, copy the initial position set inside the PF

    cmd_vel = np.array(
        [0.8, 0.03]
    )  # velocity command (v, omega). In this case will be constant for the whole simulation

    # convert the durations to number of time steps
    steps = int(sim_length_s / sim_step_s)
    pf_step = int(pf_dt / sim_step_s)
    particles_plot_step = int(particles_plot_step_s / sim_step_s)

    # Initialize a plot and insert the landmarks
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    lmarks_legend = ax[0].scatter(landmarks[:, 0], landmarks[:, 1], marker="s", s=60, label="Landmarks")

    track = []  # list to store all the robot positions
    track_odom = []  # list to store all the odometry positions
    track_pf = [pf.mu.copy()]  # list to store all the pf positions

    odom_pos_prev = odom_pos.copy()

    # plot initial distribution of particles
    init_particles_lgnd = plot_initial_particles(pf.N, pf.particles, ax=ax[0])

    # The main loop that runs the simulation
    for i in range(steps):
        if np.any(cmd_vel == 0.0):
            cmd_vel += 1e-9
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
            for lmark in landmarks:
                z = z_landm_sensor(
                    sim_pos, lmark, sigma_z, max_range=max_range, fov=fov
                )  # landmarks out of the sensor's FOV will be not detected

                # if any landmark detected by the sensor, update the PF
                if z is not None:
                    # run the correction step of the PF
                    pf.update(z, sigma_z, eval_hx=eval_hx_landm, hx_args=(lmark, sigma_z))

            # after the update of the weights with the measurements, we normalize the weights to make them probabilities
            pf.normalize_weights()

            # resample if too few effective particles
            neff = pf.neff()

            if neff < pf.N / 2:
                pf.resampling(
                    resampling_fn=pf.resampling_fn,  # simple, residual, stratified, systematic
                    resampling_args=(pf.weights,),  # tuple: only pf.weights if using pre-defined functions
                )
                assert np.allclose(pf.weights, 1 / pf.N)

            # estimate robot mean and covariance from particles
            pf.estimate(mean_fn=state_mean, residual_fn=residual, angle_idx=2)

            # plot the posterior particles every particles_plot_step seconds
            if i % particles_plot_step == 0:
                legend_PF1, legend_PF2 = plot_particles(pf.particles, sim_pos, pf.mu, ax=ax[0])
            track_pf.append(pf.mu.copy())

            #print("Step: ", i, " - NEFF: ", neff)

    # draw plots
    track = np.array(track)
    track_odom = np.array(track_odom)
    track_pf = np.array(track_pf)

    # trajectory plots
    (track_legend,) = ax[0].plot(track[:, 0], track[:, 1], label="True robot path")
    (track_odom_legend,) = ax[0].plot(track_odom[:, 0], track_odom[:, 1], "--", label="Odometry path")
    ax[0].axis("equal")
    ax[0].set_title("PF Robot localization")
    ax[0].legend(handles=[lmarks_legend, track_legend, track_odom_legend, legend_PF1, legend_PF2, init_particles_lgnd])

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
        [[5, 12], [10.5, 7.5], [16.5, 15], [10, 14], [5, 6], [14.5, 11.5], [14, 9], [8, 15.5], [13.5, 17], [18.4, 18]]
    )
    # sensor params
    max_range = 8.0
    fov = math.pi / 2

    # sim params
    pf_dt = 1.0  # time interval between measurements [s]
    sim_length_s = 22  # length of the simulation [s]

    # Probabilistic models parameters
    dim_x = 3
    # First, choose the Motion Model
    motion_model = "velocity"  # 'odometry' or 'velocity'

    # general noise parameters
    std_lin_vel = 0.1  # [m/s]
    std_ang_vel = np.deg2rad(1.0)  # [rad/s]
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
    sigma_z = np.array([std_range, std_bearing])

    # Initialize the PF
    pf = RobotPF(
        dim_x=dim_x,
        dim_u=dim_u,
        eval_gux=eval_gux,
        resampling_fn=simple_resample,
        boundaries=[(0.0, 20.0), (0.0, 20.0), (-np.pi, np.pi)],
        N=1000,
    )

    pf.mu = np.array([1, 6, 0.3])  # x, y, theta
    pf.Sigma = np.diag([0.1, 0.1, 0.1])
    pf.initialize_particles()

    run_localization_sim(
        pf,
        pf_dt=pf_dt,
        landmarks=landmarks,
        z_landm_sensor=landmark_range_bearing_sensor,
        max_range=max_range,
        fov=fov,
        eval_hx_landm=landmark_range_bearing_model,
        sigma_u=sigma_u,
        sigma_z=sigma_z,
        motion_model=motion_model,
        sigma_u_odom=sigma_u_odom,
        particles_plot_step_s=3.0,
        sim_length_s=sim_length_s,
    )

    plt.close("all")


if __name__ == "__main__":
    main()
