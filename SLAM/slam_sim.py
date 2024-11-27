import sys

sys.path.append("..")

from Gaussian_Filters.probabilistic_models import (
    landmark_range_bearing_model,
    landmark_range_bearing_sensor,
    landmark_slam_jacobian_sympy,
)
from Gaussian_Filters.utils import normalize_angle
from robot_sim import RobotSimulator
from slam_ekf import SLAM_EKF

import numpy as np
import matplotlib.pyplot as plt


def landmark_fn(x, idx, R):
    """
    Wrapper function to adapt the landmark_range_bearing_model to the EKF update function call.

    Parameters
    ----------
    x : np.array
        State vector. It is passed by default by the EKF update function.
    idx : int
        Index of the landmark in the state vector.
    R : np.array
        Measurement noise covariance matrix.

    Returns
    -------
    np.array
        Measurement prediction.
    """
    return landmark_range_bearing_model(x[0:3], x[idx : idx + 2], np.sqrt(np.diag(R)))


eval_landmark_jac = landmark_slam_jacobian_sympy()


def landmark_jacobian(x, idx):
    """
    Compute the Jacobian of the landmark sensor model.

    Parameters
    ----------
    x : np.array
        State vector. It is passed by default by the EKF update function.
    idx : int
        Index of the landmark in the state vector.

    Returns
    -------
    np.array
        Jacobian matrix.
    """
    H_low_dim = eval_landmark_jac(x[0:3], x[idx], x[idx + 1])
    H = np.zeros((2, len(x)))
    H[:, :3] = H_low_dim[:, :3]
    H[:, idx : idx + 2] = H_low_dim[:, 3:]
    return H


def residual(a, b):
    y = a - b
    y[1] = normalize_angle(y[1])
    return y


# Simulation Parameters
max_duration_s = 10  # duration of the simulation [s]
dt = 1 / 20  # time step of the simulation [s]
v0 = 1.0  # initial linear velocity [m/s]
w0 = 0.1  # initial angular velocity [rad/s]
landmarks = {
    0: np.array([2, -0.3]),
    3: np.array([1.5, 0.5]),
    4: np.array([3.0, 0.2]),
    5: np.array([4.2, 0.7]),
    2: np.array([3.1, 1.8]),
}  # id and position of the landmarks
R = np.diag([0.01, np.deg2rad(1.0)]) ** 2  # measurement noise matrix

# EKF Initialization
ekf = SLAM_EKF(dim_x=3, dim_z=2, dim_u=2, landmarks_id=[0, 1, 5, 3, 4, 2])
# ekf.x[0:3] = np.array([1, 6, 0.0])  # initial state
ekf.Mt = np.diag([0.1, np.deg2rad(1.0)]) ** 2  # noise on the control input
ekf.Q = np.eye(ekf.dim_x) * 1e-6  # process noise

# Saved data
trajectory = []

# Simulation
steps = int(max_duration_s // dt)
robot = RobotSimulator()
trajectory.append(ekf.x[:3].copy())
for i in range(steps):
    # Update command velocity
    v = v0 * (1 - i / steps)
    w = w0 + w0 * (1 - i / steps)

    # Move the robot in the simulation
    robot.move(v, w, dt)

    # Run EKF
    ekf.predict([v, w, dt])
    for lmark_id, lmark in landmarks.items():
        z = landmark_range_bearing_sensor(robot.get_pose(), lmark, np.sqrt(np.diag(R)), fov=np.pi)

        if z is not None:
            idx = ekf.map_lmark_id_to_state_idx(lmark_id)

            # TODO manage the case when the landmark is not in the list and extend the state

            # If it is the first time we see the landmark, initialize it
            if np.any(np.isnan(ekf.x[idx : idx + 2])):
                ekf.initialize_landmark(z, idx)
            else:
                ekf.update(
                    z,
                    HJacobian=landmark_jacobian,
                    args=idx,
                    Hx=landmark_fn,
                    hx_args=(idx, R),
                    R=R,
                    residual=residual,
                )

    trajectory.append(ekf.x[:3].copy())

# Plot results
x = ekf.x
P = ekf.P

print(f"Starting error {np.linalg.norm(np.array(trajectory[0]) - np.array(robot.trajectory[0])):.2f}")
for lmark_id, lmark_pos in landmarks.items():
    idx = ekf.map_lmark_id_to_state_idx(lmark_id)
    print(f"Landmark_id: {lmark_id}, Error: {np.linalg.norm(lmark_pos - x[idx:idx+2]):.2f}")

plt.figure()
plt.plot([x[0] for x in robot.trajectory], [x[1] for x in robot.trajectory], label="True trajectory")
plt.plot([x[0] for x in trajectory], [x[1] for x in trajectory], label="Estimated trajectory", c="g")
plt.scatter([x[0] for x in landmarks.values()], [x[1] for x in landmarks.values()], label="Landmarks", c="r")
plt.scatter(x[3::2], x[4::2], label="Estimated landmarks", c="g")
plt.axis("equal")
plt.legend()
plt.show()
