import math
from math import cos, sin, sqrt
import numpy as np
import sympy
from sympy import symbols, Matrix


def sample_velocity_motion_model(x, u, a, dt):
    """ Sample velocity motion model.
    Arguments:
    x -- pose of the robot before moving [x, y, theta]
    u -- velocity reading obtained from the robot [v, w]
    sigma -- noise parameters of the motion model [a1, a2, a3, a4, a5, a6] or [std_dev_v, std_dev_w]
    dt -- time interval of prediction
    """

    sigma = np.ones((3))
    if a.shape == u.shape:
        sigma[:-1] = a[:]
        sigma[-1] = a[1]*0.5
    else:
        sigma[0] = a[0]*u[0]**2 + a[1]*u[1]**2
        sigma[1] = a[2]*u[0]**2 + a[3]*u[1]**2
        sigma[2] = a[4]*u[0]**2 + a[5]*u[1]**2

    # sample noisy velocity commands to consider actuaction errors and unmodeled dynamics
    v_hat = u[0] + np.random.normal(0, sigma[0])
    w_hat = u[1] + np.random.normal(0, sigma[1])
    gamma_hat = np.random.normal(0, sigma[2])

    # compute the new pose of the robot according to the velocity motion model
    r = v_hat/w_hat
    x_prime = x[0] - r*sin(x[2]) + r*sin(x[2]+w_hat*dt)
    y_prime = x[1] + r*cos(x[2]) - r*cos(x[2]+w_hat*dt)
    theta_prime = x[2] + w_hat*dt + gamma_hat*dt

    return np.array([x_prime, y_prime, theta_prime])

def get_odometry_command(odom_pose, odom_pose_prev):
    """ Transform robot poses taken from odometry to u command
    Arguments:
    odom_pose -- last odometry pose of the robot [x, y, theta] at time t
    odom_pose_prev -- previous odometry pose of the robot [x, y, theta] at time t-1

    Output: 
    u_odom : np.array [rot1, trasl, rot2]
    """

    x_odom, y_odom, theta_odom = odom_pose[:]
    x_odom_prev, y_odom_prev, theta_odom_prev = odom_pose_prev[:]

    rot1 = math.atan2(y_odom - y_odom_prev, x_odom - x_odom_prev) - theta_odom_prev
    trasl = sqrt((x_odom - x_odom_prev) ** 2 + (y_odom - y_odom_prev) ** 2)
    rot2 = theta_odom - theta_odom_prev - rot1

    return np.array([rot1, trasl, rot2])

def sample_odometry_motion_model(x, u, a):
    """ Sample odometry motion model.
    Arguments:
    x -- pose of the robot before moving [x, y, theta]
    u -- odometry reading obtained from the robot [rot1, trans, rot2]
    a -- noise parameters of the motion model [a1, a2, a3, a4] or [std_rot1, std_trans, std_rot2]
    """

    sigma = np.ones((3))
    if a.shape == u.shape:
        sigma = a
    else:
        sigma[0] = a[0]*abs(u[0]) + a[1]*abs(u[1])
        sigma[1] = a[2]*abs(u[1]) + a[3]*(abs(u[0])+abs(u[2]))
        sigma[2] = a[0]*abs(u[2]) + a[1]*abs(u[1])
    
    # noisy odometric transformations: 1 translation and 2 rotations
    delta_hat_r1 = u[0] + np.random.normal(0, sigma[0])
    delta_hat_t = u[1] + np.random.normal(0, sigma[1])
    delta_hat_r2 = u[2] + np.random.normal(0, sigma[2])

    # new pose predicted for the robot
    x_prime = x[0] + delta_hat_t * math.cos(x[2] + delta_hat_r1)
    y_prime = x[1] + delta_hat_t * math.sin(x[2] + delta_hat_r1)
    theta_prime = x[2] + delta_hat_r1 + delta_hat_r2

    return np.array([x_prime, y_prime, theta_prime])


def landmark_range_bearing_model(robot_pose, landmark, sigma):
    """""
    Sampling z from landmark model for range and bearing
    """""
    m_x, m_y = landmark[:]
    x, y, theta = robot_pose[:]

    r_ = math.dist([x, y], [m_x, m_y]) + np.random.normal(0., sigma[0])
    phi_ = math.atan2(m_y - y, m_x - x) - theta + np.random.normal(0., sigma[1])
    return np.array([r_, phi_])


def landmark_range_bearing_sensor(robot_pose, landmark, sigma, max_range=6.0, fov=math.pi/2):
    """""
    Simulate the detection of a landmark with a virtual sensor able to estimate range and bearing
    """""
    z = landmark_range_bearing_model(robot_pose, landmark, sigma)

    # filter z for a more realistic sensor model (add a max range distance and a FOV)
    if z[0] > max_range or abs(z[1]) > fov / 2:
        return None

    return z

def velocity_mm_Gt(x, u, dt):
    """
    Evaluate Jacobian Gt w.r.t state x=[x, y, theta]
    """
    theta = x[2]
    v, w = u[0], u[1]
    r = v/w
    Gt = np.array([
        [1, 0, -r*cos(theta) + r*cos(theta+w*dt)],
        [0, 1, -r*sin(theta) + r*sin(theta+w*dt)],
        [0, 0, 1]
    ])

    return Gt

def velocity_mm_Vt(x, u, dt):
    """
    Evaluate Jacobian Vt w.r.t command u=[v,w]
    """
    theta = x[2]
    v, w = u[0], u[1]
    r = v/w
    Vt = np.array([
        [-sin(theta)/w + sin(theta+w*dt)/w, 
         dt*v*cos(theta+w*dt)/w + v*sin(theta)/w**2 - v*sin(theta+w*dt)/w**2],
        [-cos(theta)/w - cos(theta+w*dt)/w, 
         dt*v*sin(theta+w*dt)/w - v*cos(theta)/w**2 + v*cos(theta+w*dt)/w**2],
        [0, dt]
    ])

    return Vt

def landmark_sm_Ht(x, mx, my):
    x, y = x[0], x[1]
    div = sqrt((mx - x)**2 + (my - y)**2)
    Ht = np.array([
        [(-mx + x)/div, (-my + y)/div, 0],
        [-(-my+y)/div, -(mx - x)/div, -1]
    ])

    return Ht

# decorator
def squeeze_sympy_out(func):
    # inner function
    def squeeze_out(*args):
        out = func(*args).squeeze()
        return out
    return squeeze_out

def velocity_mm_simpy():
    """
    Define Jacobian Gt w.r.t state x=[x, y, theta] and Vt w.r.t command u=[v, w]
    """
    x, y, theta, v, w, dt = symbols('x y theta v w dt')
    R = v / w
    beta = theta + w * dt
    gux = Matrix(
        [
            [x - R * sympy.sin(theta) + R * sympy.sin(beta)],
            [y + R * sympy.cos(theta) - R * sympy.cos(beta)],
            [beta],
        ]
    )

    eval_gux = squeeze_sympy_out(sympy.lambdify((x, y, theta, v, w, dt), gux, 'numpy'))

    Gt = gux.jacobian(Matrix([x, y, theta]))
    eval_Gt = squeeze_sympy_out(sympy.lambdify((x, y, theta, v, w, dt), Gt, "numpy"))
    #print("Gt:", Gt)

    Vt = gux.jacobian(Matrix([v, w]))
    eval_Vt = squeeze_sympy_out(sympy.lambdify((x, y, theta, v, w, dt), Vt, "numpy"))
    #print("Vt:", Vt)

    return eval_gux, eval_Gt, eval_Vt

def odometry_mm_simpy():
    """
    Define Jacobian Gt and Vt for the odometry motion model
    """
    rot1, trasl, rot2 = symbols(r"\delta_{rot1} \delta_{trasl} \delta_{rot2}")
    x, y, theta = symbols(r"x y \theta")
    gux_odom = Matrix([
        [x + trasl * sympy.cos(theta + rot1)],
        [y + trasl * sympy.sin(theta + rot1)],
        [theta + rot1 + rot2],
    ])
    Gt_odom = gux_odom.jacobian(Matrix([x, y, theta]))
    Vt_odom = gux_odom.jacobian(Matrix([rot1, trasl, rot2]))

    args = (x, y, theta, rot1, trasl, rot2)
    eval_gux_odom = squeeze_sympy_out(sympy.lambdify(args, gux_odom, "numpy"))
    eval_Gt_odom = squeeze_sympy_out(sympy.lambdify(args, Gt_odom, "numpy"))
    eval_Vt_odom = squeeze_sympy_out(sympy.lambdify(args, Vt_odom, "numpy"))

    return eval_gux_odom, eval_Gt_odom, eval_Vt_odom

def landmark_sm_simpy():
    x, y, theta, mx, my = symbols("x y theta m_x m_y")

    hx = Matrix(
        [
            [sympy.sqrt((mx - x) ** 2 + (my - y) ** 2)],
            [sympy.atan2(my - y, mx - x) - theta],
        ]
    )
    eval_hx = squeeze_sympy_out(sympy.lambdify((x, y, theta, mx, my), hx, "numpy"))
    
    Ht = hx.jacobian(Matrix([x, y, theta]))
    eval_Ht = squeeze_sympy_out(sympy.lambdify((x, y, theta, mx, my), Ht, "numpy"))
    #print("Ht:", Ht)

    return eval_hx, eval_Ht

