# Generated from Motion_Models/velocity_motion_model.py; do not edit.
# Source SHA256: 388c3e3730acf8d6787a3a97bd3055bd61ec59a7fbc126acea29f6db244d1f5c
# Repository license applies; see LICENSE at the repository root.
import numpy as np
from math import sin, cos


def propagate_velocity(x, v_hat, w_hat, gamma_hat, dt):
    """Same circular-arc model, with the analytic straight-line limit.

    sin(half_turn)/half_turn avoids cancellation/division by zero at omega=0.
    """
    half_turn = 0.5 * w_hat * dt
    sinc = 1.0 if abs(half_turn) < 1e-10 else sin(half_turn) / half_turn
    travel = v_hat * dt * sinc
    return np.array([
        x[0] + travel * cos(x[2] + half_turn),
        x[1] + travel * sin(x[2] + half_turn),
        x[2] + (w_hat + gamma_hat) * dt,
    ])


def sample_velocity_motion_model(x, u, a, dt):
    v_hat = u[0] + np.random.normal(0, a[0] * u[0] ** 2 + a[1] * u[1] ** 2)
    w_hat = u[1] + np.random.normal(0, a[2] * u[0] ** 2 + a[3] * u[1] ** 2)
    gamma_hat = np.random.normal(0, a[4] * u[0] ** 2 + a[5] * u[1] ** 2)
    return propagate_velocity(x, v_hat, w_hat, gamma_hat, dt)
