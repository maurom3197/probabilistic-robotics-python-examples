import sys

sys.path.append("..")
from Gaussian_Filters.probabilistic_models import sample_velocity_motion_model

import numpy as np


class RobotSimulator:
    """
    A class to simulate a robot moving in a 2D plane with a differential drive model.
    """

    def __init__(self, x=0, y=0, theta=0, cmd_vel_std=(0.1, np.deg2rad(1.0))):
        """
        Parameters
        ----------
        x : float
            Initial x position
        y : float
            Initial y position
        theta : float
            Initial orientation
        cmd_vel_std : tuple
            Standard deviation of the noise in velocity command
        """
        self.x = x
        self.y = y
        self.theta = theta

        self.x_odom = 0.0
        self.y_odom = 0.0
        self.theta_odom = 0.0

        self.noise = np.array(cmd_vel_std)
        self.trajectory = [(x, y, theta)]

    def move(self, v, w, dt):
        """
        Move the robot in the simulation and update the odometry.
        Parameters
        ----------
        v : float
            Linear velocity
        w : float
            Angular velocity
        dt : float
            Time step
        """
        self.x, self.y, self.theta = sample_velocity_motion_model(
            (self.x, self.y, self.theta), np.array([v, w]), self.noise, dt
        )
        self.__odometry(v, w, dt)
        self.trajectory.append((self.x, self.y, self.theta))

    def get_pose(self):
        return self.x, self.y, self.theta

    def get_odom(self):
        return self.x_odom, self.y_odom, self.theta_odom

    def __odometry(self, v, w, dt):
        self.x_odom, self.y_odom, self.theta_odom = sample_velocity_motion_model(
            (self.x_odom, self.y_odom, self.theta_odom), np.array([v, w]), self.noise, dt
        )
