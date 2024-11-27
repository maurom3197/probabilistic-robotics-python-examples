import sys

sys.path.append("..")

from Gaussian_Filters.probabilistic_models import (
    sample_velocity_motion_model,
    velocity_mm_Gt,
    velocity_mm_Vt,
)
from filterpy.kalman import ExtendedKalmanFilter

import numpy as np


class SLAM_EKF(ExtendedKalmanFilter):

    def __init__(self, dim_x, dim_z, dim_u=2, landmarks_id=[]):
        """
        Parameters
        ----------
        dim_x : int
            Dimension of the state vector, without considering the landmarks

        dim_z : int
            Same as in the parent class

        dim_u : int
            Same as in the parent class

        landmarks_id : list
            List of the known landmarks ids
        """
        # Initialize landmarks
        self.n_lmarks = len(landmarks_id)
        self.landmarks_id = landmarks_id

        # Initialize the parent class
        super().__init__((dim_x + 2 * self.n_lmarks), dim_z, dim_u)

        # State initialization
        self.x = np.hstack(([1e-9, 1e-9, 1e-9], np.ones(2 * self.n_lmarks) * np.nan))
        self.P = np.zeros((self.dim_x, self.dim_x))
        self.P[0:3, 0:3] = np.eye(3) * 1e-9
        self.P[3:, 3:] = np.eye(2 * self.n_lmarks)

        # Noise parameters
        self.Mt = np.eye(dim_u)  # to be set by the user when creating the object
        self.Q *= 1e-4  # to be set by the user when creating the object

    def map_lmark_id_to_state_idx(self, landmark_id):
        if landmark_id not in self.landmarks_id:
            # TODO manage the case when the landmark is not in the list and extend the state
            raise ValueError(f"Unknown landmark id: {landmark_id}")
        else:
            return 3 + 2 * self.landmarks_id.index(landmark_id)
        
    def initialize_landmark(self, z, idx):
        self.x[idx] = self.x[0] + z[0] * np.cos(z[1] + self.x[2])
        self.x[idx + 1] = self.x[1] + z[0] * np.sin(z[1] + self.x[2])


def predict_x_slam(self: SLAM_EKF, u):
    """
    Predict the state of the system using the motion model and update the state transition matrix.
    Parameters
    ----------
    u : np.array
        Control input [v, w, dt]
    """
    cmd_vel = np.array(u[0:2])
    dt = u[-1]
    # Predict the state
    self.x[:3] = sample_velocity_motion_model(self.x[:3], cmd_vel, np.diag(self.Mt), dt)
    self.F[0:3, 0:3] = velocity_mm_Gt(self.x[:3], cmd_vel, dt)
    V = velocity_mm_Vt(self.x[:3], cmd_vel, dt)
    self.Q[0:3, 0:3] = V @ self.Mt @ V.T


SLAM_EKF.predict_x = predict_x_slam
