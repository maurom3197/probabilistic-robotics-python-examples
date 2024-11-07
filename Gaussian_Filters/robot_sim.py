import math
import numpy as np

from utils import residual
from plot_utils import plot_covariance
import matplotlib.pyplot as plt
from ekf import RobotEKF
from probabilistic_models import sample_velocity_motion_model, sample_odometry_motion_model, get_odometry_command,landmark_range_bearing_model, landmark_range_bearing_sensor
from probabilistic_models import velocity_mm_simpy, odometry_mm_simpy, landmark_sm_simpy


def run_localization_sim(
    ekf: RobotEKF,
    ekf_dt,
    landmarks,
    max_range,
    fov,
    z_landm_sensor,
    eval_hx_landm,
    eval_Ht_landm,
    Q_landm,
    sigma_u,
    sigma_z,
    motion_model = 'velocity',
    sigma_u_odom = 0.,
    sim_step_s=0.1,
    ellipse_step_s=5.0,
    sim_length_s=1,
    constant_vel=True,
    ):

    sim_pos = ekf.mu.copy()  # simulated position, copy the initial position set inside the EKF
    odom_pos = ekf.mu.copy()  # odometry position, copy the initial position set inside the EKF
    sim_noise_generator = np.random.default_rng(42424242)  # random noise generator

    cmd_vel = np.array(
        [1.10, 0.05]
    )   # velocity command (v, omega). In this case will be constant for the whole simulation

    # convert the durations to number of time steps
    steps = int(sim_length_s / sim_step_s)
    ekf_step = int(ekf_dt / sim_step_s)
    ellipse_step = int(ellipse_step_s / sim_step_s)

    # Initialize a plot and insert the landmarks
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    lmarks_legend = ax[0].scatter(landmarks[:, 0], landmarks[:, 1], marker="s", s=60, label="Landmarks")

    track = []  # list to store all the robot positions
    track_odom = []  # list to store all the odometry positions
    track_ekf = [ekf.mu.copy()]  # list to store all the ekf positions

    odom_pos_prev = odom_pos.copy()

    # The main loop that runs the simulation
    for i in range(steps):
        if not constant_vel:
            if i % 10 == 0:
                cmd_vel = np.array([1.0, -0.01])
            else:
                cmd_vel = np.array([1.10, 0.03])

        # Simulate robot motion for sim_step_s seconds using the Motion Model.
        # the sampling motion model already include Gaussian noise on the command
        sim_pos = sample_velocity_motion_model(sim_pos, cmd_vel, sigma_u, sim_step_s) 
        track.append(sim_pos)

        # to simulate the error in the odometry reading, we take another Gaussian sample of the velocity command
        #noisy_vel = cmd_vel + sigma_u * sim_noise_generator.normal(size=cmd_vel.shape)
        odom_pos = sample_velocity_motion_model(odom_pos, cmd_vel, sigma_u, sim_step_s)
        track_odom.append(odom_pos)

        if i % ekf_step == 0 and i != 0:  # only update ekf at dt intervals
            # run the prediction step of the EKF
            if motion_model == 'velocity':
                ekf.predict(u=cmd_vel, sigma_u=sigma_u, g_extra_args=(ekf_dt,))
            elif motion_model == 'odometry':
                u = get_odometry_command(odom_pos, odom_pos_prev)
                ekf.predict(u=u, sigma_u=sigma_u_odom)
                odom_pos_prev = odom_pos.copy()

            # plot the prior covariance ellipses every ellipse_step_s seconds
            if i % ellipse_step == 0:  
                pri_ellipse = plot_covariance(
                    (ekf.mu[0], ekf.mu[1]),
                    ekf.Sigma[0:2, 0:2],
                    std=6,
                    facecolor="k",
                    alpha=0.4,
                    label="Predicted Cov",
                    ax=ax[0],
                )

            # for each landmark simulate the measurement of the landmark
            for lmark in landmarks:  
                z = z_landm_sensor(sim_pos, lmark, sigma_z, max_range=max_range, fov=fov)  # landmarks out of the sensor's FOV will be not detected

                # if any landmark detected by the sensor, update the EKF
                if z is not None:
                    # run the correction step of the EKF
                    ekf.update(z, eval_hx=eval_hx_landm, eval_Ht=eval_Ht_landm, Qt=Q_landm, 
                               Ht_args=(*ekf.mu, *lmark), # the Ht function requires a flattened array of parameters
                               hx_args=(ekf.mu, lmark, sigma_z),
                               residual=residual, 
                               angle_idx=-1)

            # plot the posterior covariance ellipses every ellipse_step_s seconds
            if i % ellipse_step == 0:  
                post_ellipse = plot_covariance(
                    (ekf.mu[0], ekf.mu[1]),
                    ekf.Sigma[0:2, 0:2],
                    std=6,
                    facecolor="g",
                    alpha=0.8,
                    label="Corrected Cov",
                    ax=ax[0],
                )
            track_ekf.append(ekf.mu.copy())
    
    # draw plots
    track = np.array(track)
    track_odom = np.array(track_odom)
    track_ekf = np.array(track_ekf)
    
    # trajectory plots
    (track_legend,) = ax[0].plot(track[:, 0], track[:, 1], label="Real robot path")
    (track_odom_legend,) = ax[0].plot(track_odom[:, 0], track_odom[:, 1], "--", label="Odometry path")
    ax[0].axis("equal")
    ax[0].set_title("EKF Robot localization")
    ax[0].legend(handles=[lmarks_legend, track_legend, track_odom_legend, pri_ellipse, post_ellipse])

    # error plots
    ekf_err, =  ax[1].plot(
        np.arange(0, sim_length_s, ekf_dt), 
        np.linalg.norm(track[::ekf_step, :2] - track_ekf[:, :2], axis=1), 
        '-o',
        label="EKF error",
    )
    odom_err, = ax[1].plot(
        np.arange(0, sim_length_s, sim_step_s), 
        np.linalg.norm(track[:, :2] - track_odom[:, :2], axis=1), 
        label="Odometry error",
    )
    ax[1].legend(handles=[ekf_err, odom_err])
    ax[1].set_title("Robot path error")

    fig.suptitle("EKF Robot localization, Velocity Motion Model")
    fig.tight_layout()

    plt.show()


def main():

    ##### Define Parameters #####
    
    seed = 42  # because it is the answer to the Ultimate Question of Life, The Universe and Everything :)
    np.random.seed(seed)  

    # landmarks list in map's coordinate
    landmarks = np.array([[5, 12], [10.5, 6], 
                          [16, 15], [10, 14], 
                          [6,6], [13.5, 11.5],
                          [17, 17.5]]) 
    # sensor params
    max_range = 8.0
    fov = math.pi/3

    # sim params
    ekf_dt = 1.0  # time interval between measurements [s]
    sim_length_s = 16  # length of the simulation [s]

    # Probabilistic models parameters
    # First, choose the Motion Model
    motion_model = 'odometry' # 'odometry' or 'velocity'

    # general noise parameters
    std_lin_vel = 0.1  # [m/s]
    std_ang_vel = np.deg2rad(1.0)  # [rad/s]
    std_range = 0.1  # [m]
    sigma_u = np.array([std_lin_vel, std_ang_vel])
    sigma_u_odom = 0

    # Velocity motion model params
    if motion_model == 'velocity':
        Mt = np.diag([std_lin_vel**2, std_ang_vel**2])
        eval_gux = sample_velocity_motion_model
        _, eval_Gt, eval_Vt = velocity_mm_simpy()
    
    # odometry motion model params
    elif motion_model == 'odometry':
        std_rot1 = np.deg2rad(1.0)
        std_transl = 0.05
        std_rot2 = np.deg2rad(0.05)
        sigma_u_odom = np.array([std_rot1, std_transl, std_rot2])
        Mt = np.diag([std_rot1**2, std_transl**2, std_rot2**2])
        eval_gux = sample_odometry_motion_model
        _, eval_Gt, eval_Vt = odometry_mm_simpy()
    
    # Define noise params and Q for landmark sensor model
    std_bearing = np.deg2rad(1.0)  # [rad]
    sigma_z = np.array([std_range, std_bearing])
    Q_landm = np.diag([std_range**2, std_bearing**2])
    # Define H Jacobian function
    _, eval_Ht = landmark_sm_simpy()

    # Initialize the EKF
    ekf = RobotEKF(
        dim_x = 3,
        dim_u = 2,
        eval_gux = eval_gux,
        eval_Gt = eval_Gt,
        eval_Vt = eval_Vt
    )
    ekf.mu = np.array([2, 6, 0.3])  # x, y, theta
    ekf.Sigma = np.diag([0.1, 0.1, 0.1])
    ekf.Mt = Mt

    run_localization_sim(
        ekf,
        ekf_dt = ekf_dt,
        landmarks = landmarks,
        z_landm_sensor = landmark_range_bearing_sensor,
        max_range = max_range,
        fov = fov,
        eval_hx_landm = landmark_range_bearing_model,
        eval_Ht_landm = eval_Ht, 
        Q_landm = Q_landm,
        sigma_u = sigma_u,
        sigma_z = sigma_z,
        motion_model = motion_model,
        sigma_u_odom = sigma_u_odom,
        ellipse_step_s = 2.0,
        sim_length_s = sim_length_s,
    )

    plt.close("all")

if __name__ == "__main__":
    main()
