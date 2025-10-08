import math
import numpy as np

from Gaussian_Filters.utils import residual, state_mean, z_mean
from Gaussian_Filters.plot_utils import plot_covariance
import matplotlib.pyplot as plt
from Gaussian_Filters.ukf import RobotUKF, SigmaPoints
from Gaussian_Filters.probabilistic_models import sample_velocity_motion_model, sample_odometry_motion_model, get_odometry_command
from Gaussian_Filters.probabilistic_models import landmark_range_bearing_model, landmark_range_bearing_sensor
from scipy.linalg import cholesky, sqrtm
from Gaussian_Filters.utils import nearestPD


def run_localization_sim(
    ukf: RobotUKF,
    ukf_dt,
    landmarks,
    max_range,
    fov,
    z_landm_sensor,
    eval_hx_landm,
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

    sim_pos = ukf.mu.copy()  # simulated position, copy the initial position set inside the UKF
    odom_pos = ukf.mu.copy()  # odometry position, copy the initial position set inside the UKF

    cmd_vel = np.array(
        [0.8, 0.03]
    )   # velocity command (v, omega). In this case will be constant for the whole simulation

    # convert the durations to number of time steps
    steps = int(sim_length_s / sim_step_s)
    ukf_pred_step = int(ukf_dt[0] / sim_step_s)
    ukf_update_step = int(ukf_dt[1] / sim_step_s)
    ellipse_step = int(ellipse_step_s / sim_step_s)

    # Initialize a plot and insert the landmarks
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    lmarks_legend = ax[0].scatter(landmarks[:, 0], landmarks[:, 1], marker="s", s=60, label="Landmarks")

    track = []  # list to store all the robot positions
    track_odom = []  # list to store all the odometry positions
    track_ukf = [ukf.mu.copy()]  # list to store all the ukf positions

    odom_pos_prev = odom_pos.copy()

    # The main loop that runs the simulation
    for i in range(steps):
        if not constant_vel:
            if i % 10 == 0:
                cmd_vel = np.array([1.0, -0.01])
            else:
                cmd_vel = np.array([1.10, 0.03])
        
        if np.any(cmd_vel == 0.):
            cmd_vel += 1e-7

        # Simulate robot motion for sim_step_s seconds using the Motion Model.
        # the sampling motion model already include Gaussian noise on the command
        sim_pos = sample_velocity_motion_model(sim_pos, cmd_vel, sigma_u, sim_step_s) 
        track.append(sim_pos)

        # to simulate the error in the odometry reading, we take another Gaussian sample of the velocity command
        #noisy_vel = cmd_vel + sigma_u * sim_noise_generator.normal(size=cmd_vel.shape)
        odom_pos = sample_velocity_motion_model(odom_pos, cmd_vel, sigma_u, sim_step_s)
        track_odom.append(odom_pos)

        if i % ukf_pred_step == 0 and i != 0:  # only update ukf at dt intervals
            # run the prediction step of the UKF
            if motion_model == 'velocity':
                ukf.predict(u=cmd_vel, sigma_u=sigma_u, g_extra_args=(ukf_dt[0],),
                            mean_fn=state_mean, residual_fn=residual, angle_idx=2)
            elif motion_model == 'odometry':
                u = get_odometry_command(odom_pos, odom_pos_prev)
                ukf.predict(u=u, sigma_u=sigma_u_odom, 
                            mean_fn=state_mean, residual_fn=residual, angle_idx=2)
                odom_pos_prev = odom_pos.copy()
            
            try:
                _ = cholesky(ukf.Sigma)
            except np.linalg.LinAlgError:
                ukf.Sigma = nearestPD(ukf.Sigma)

            # plot the prior covariance ellipses every ellipse_step_s seconds
            if i % ellipse_step == 0:  
                pri_ellipse = plot_covariance(
                    (ukf.mu[0], ukf.mu[1]),
                    ukf.Sigma[0:2, 0:2],
                    std=6,
                    facecolor="k",
                    alpha=0.4,
                    label="Predicted Cov",
                    ax=ax[0],
                )

        if i % ukf_update_step == 0 and i != 0:  # only update ukf at dt intervals
            # for each landmark simulate the measurement of the landmark
            for lmark in landmarks:  
                z = z_landm_sensor(sim_pos, lmark, sigma_z, max_range=max_range, fov=fov)  
                # landmarks out of the sensor's FOV will be not detected

                # if any landmark detected by the sensor, update the UKF
                if z is not None:
                    # run the correction step of the UKF
                    ukf.update(z, eval_hx=eval_hx_landm, Qt=Q_landm, 
                               hx_args=(lmark, sigma_z),
                               mean_fn=z_mean,
                               residual_fn=residual, 
                               angle_idx=-1)
                    try:
                        _ = cholesky(ukf.Sigma)
                    except np.linalg.LinAlgError:
                        ukf.Sigma = nearestPD(ukf.Sigma)

            # plot the posterior covariance ellipses every ellipse_step_s seconds
            if i % ellipse_step == 0:  
                post_ellipse = plot_covariance(
                    (ukf.mu[0], ukf.mu[1]),
                    ukf.Sigma[0:2, 0:2],
                    std=6,
                    facecolor="g",
                    alpha=0.8,
                    label="Corrected Cov",
                    ax=ax[0],
                )
            track_ukf.append(ukf.mu.copy())
    
    # draw plots
    track = np.array(track)
    track_odom = np.array(track_odom)
    track_ukf = np.array(track_ukf)
    
    # trajectory plots
    (track_legend,) = ax[0].plot(track[:, 0], track[:, 1], label="Real robot path")
    (track_odom_legend,) = ax[0].plot(track_odom[:, 0], track_odom[:, 1], "--", label="Odometry path")
    ax[0].axis("equal")
    ax[0].set_title("UKF Robot localization")
    ax[0].legend(handles=[lmarks_legend, track_legend, track_odom_legend, pri_ellipse, post_ellipse])

    # error plots
    ukf_err, =  ax[1].plot(
        np.arange(0, sim_length_s, ukf_dt[1]), 
        np.linalg.norm(track[::ukf_update_step, :2] - track_ukf[:, :2], axis=1), 
        '-o',
        label="UKF error",
    )
    odom_err, = ax[1].plot(
        np.arange(0, sim_length_s, sim_step_s), 
        np.linalg.norm(track[:, :2] - track_odom[:, :2], axis=1), 
        label="Odometry error",
    )
    ax[1].legend(handles=[ukf_err, odom_err])
    ax[1].set_title("Robot path error")

    fig.suptitle("UKF Robot localization, Velocity Motion Model")
    fig.tight_layout()

    plt.show()


def main():

    ##### Define Parameters #####
    
    seed = 42  # because it is the answer to the Ultimate Question of Life, The Universe and Everything :)
    np.random.seed(seed)  

    # landmarks list in map's coordinate
    landmarks = np.array([[5, 12], [10.5, 7.5], 
                          [16.5, 15], [10, 14], 
                          [5,6], [14.5, 11.5],
                          [13, 9], [8, 15.5],
                          [13.5, 17], [18.4, 18]]) 
    # sensor params
    max_range = 8.0
    fov = math.pi/3

    # sim params
    ukf_dt = [1.0, 1.0]  # time interval between measurements [s]
    sim_length_s = 22 # length of the simulation [s]

    # Probabilistic models parameters
    dim_x = 3
    # First, choose the Motion Model
    motion_model = 'odometry' # 'odometry' or 'velocity'

    # general noise parameters
    std_lin_vel = 0.01  # [m/s]
    std_ang_vel = np.deg2rad(0.3)  # [rad/s]
    sigma_u = np.array([std_lin_vel, std_ang_vel])
    sigma_u_odom = 0

    # Velocity motion model params
    if motion_model == 'velocity':
        dim_u = 2
        eval_gux = sample_velocity_motion_model
    
    # odometry motion model params
    elif motion_model == 'odometry':
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
    Q_landm = np.diag([std_range**2, std_bearing**2])

    # Define sigma points function
    sigma_points = SigmaPoints(
        n=dim_x, 
        alpha=.6, 
        beta=2., 
        kappa=0, # 3 - dim(x)
        sqrt=cholesky,
        subtract=residual
        )

    # Initialize the UKF
    ukf = RobotUKF(
        dim_x = dim_x,
        dim_u = dim_u,
        dim_z=2,
        points=sigma_points,
        eval_gux = eval_gux,
    )
    ukf.mu = np.array([1, 6, 0.3])  # x, y, theta
    ukf.Sigma = np.eye(dim_x)*0.01
    ukf.R = np.eye(dim_x)*0.001

    run_localization_sim(
        ukf,
        ukf_dt = ukf_dt,
        landmarks = landmarks,
        z_landm_sensor = landmark_range_bearing_sensor,
        max_range = max_range,
        fov = fov,
        eval_hx_landm = landmark_range_bearing_model,
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
