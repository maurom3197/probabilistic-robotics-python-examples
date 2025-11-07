[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) 

<h1 align="center">  probabilistic-robotics-python-examples
</h1>


<p align="center">
  <img src=./readme_images/landmark_model_sampling.png alt="Alternative text" width="400">
  <img src=./readme_images/multi_odometry_samples.png alt="Alternative text" width="400"/>
</p>

This repository contains Python script and notebooks to reproduce practical examples of popular algorithms presented in the Probabilistic Robotics book for didactic activities.

# Content organization (work in progress)
  - Motion Models:
    - Odometry Motion Model
    - Velocity Motion Model
  - Sensors Models:
    - Beam Range Model 
    - Likelihood Fields
    - Landmark Model
    - utils algorithms: ray casting, grid map utils, generate beam data
  - Gaussian Filters:
    - Extended Kalman Filter: ekf, ekf_robot_sim
    - Unscented Kalman Filter: ukf, ukf_robot_sim (TO DO: improve numerical stability)
    - probabilistic models: 
    - utils: utility functions (residual, mean, metrics), plot_utils

<p align="center">
  <img src=./readme_images/expected_output_odom.png alt="Alternative text" width="650"/>
</p>

# References
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
