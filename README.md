[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) 

<h1 align="center">  probabilistic-robotics-python-examples
</h1>


<p align="center">
  <img src=./readme_images/landmark_model_sampling.png alt="Samples from the landmark sensor model" width="400">
  <img src=./readme_images/multi_odometry_samples.png alt="Odometry samples over multiple motion steps" width="400"/>
</p>

Python scripts and a notebook for practical, educational examples of algorithms from *Probabilistic Robotics*. The examples simulate robot motion, measurements, localization, mapping, and SLAM.

## Repository organization

| Directory | Contents |
| --- | --- |
| `Motion_Models/` | Odometry and velocity motion models |
| `Sensors_Models/` | Beam range and landmark models, likelihood fields, ray casting, and data generation |
| `Gaussian_Filters/` | EKF and UKF implementations, localization examples, and the Jacobian notebook |
| `Discrete_Filters/` | Particle filter implementation and localization examples |
| `Mapping/` | Grid-map utilities, occupancy grid mapping from know poses, MAP mapping |
| `SLAM/` | EKF SLAM implementation and simulation|
| `2D_maps/` | Sample map images |

<p align="center">
  <img src=./readme_images/expected_output_odom.png alt="Expected odometry motion model output" width="650"/>
</p>

## Setup
These instructions assume a working Ubuntu environment with Git, Python 3, and venv support installed.
### 1. Clone the repository’s main branch and enter its root directory:

```bash
git clone --branch main https://github.com/maurom3197/probabilistic-robotics-python-examples.git
cd probabilistic-robotics-python-examples
```

### 2. Create and activate a virtual environment

On Linux, with Python 3 installed:

Create the venv called 'prob-robotics-python'
```bash
python3 -m venv prob-robotics-python
```
Activate it:
```bash
source prob-robotics-python/bin/activate
```
Activate the environment again whenever you open a new terminal. Run `deactivate` when you are finished.

### 3. Install dependencies and the local packages
Install both the dependencies and the local packages with:
```bash
pip install -e .
```

Dependencies include NumPy and SciPy for numerical algorithms, Matplotlib for plots, Pillow and scikit-image for maps, SymPy for symbolic Jacobians, FilterPy for SLAM, scikit-learn, and ipykernel. See [requirements.txt](requirements.txt) for the exact pins.

## Running the examples

Run these commands from the repository root with the virtual environment active. Example: Use `python -m package.module.py` so imports between the example packages resolve correctly.

When opening a new terminal, return to the repository root and activate the environment before running scripts:

```bash
cd probabilistic-robotics-python-examples
source prob-robotics-python/bin/activate
python -m Motion_Models.velocity_motion_model
```

## References
```
@book{10.5555/1121596,
author = {Thrun, Sebastian and Burgard, Wolfram and Fox, Dieter},
title = {Probabilistic Robotics (Intelligent Robotics and Autonomous Agents)},
year = {2005},
isbn = {0262201623},
publisher = {The MIT Press}
}
```

# Acknowledgements
This work has been realized thanks to a joint effort by researchers at PIC4SeR Centre for Service Robotics at Politecnico di Torino (https://pic4ser.polito.it/). It supports the didactic activity of the course ([Sensors, embedded systems and algorithms for Service Robotics](https://didattica.polito.it/pls/portal30/gap.pkg_guide.viewGap?p_cod_ins=01HFWQW&p_a_acc=2025&p_header=S&p_lang=IT&multi=N)) offered from 2023/24 in the M.Sc. in Mechatronic Engineering at Politecnico di Torino. 
