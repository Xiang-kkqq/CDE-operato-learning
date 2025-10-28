# CDE-operato-learning

1. data/data_stewartv13
    This dataset is designed for training and validating data-driven forward kinematics models of a 6-6 Stewart platform.
    Each sample in the dataset represents a valid configuration of the parallel robot, consisting of six given actuator lengths and the corresponding end-effector pose expressed in screw parameters.

    The data are stored in MATLAB .mat format, with files named T1.mat to T8.mat.
    Each file contains a variable Tn, which is a numerical matrix with 13 columns.

   The first six columns correspond to the actuator lengths of the six legs:L = [l1, l2, l3, l4, l5, l6], representing the input driving variables of the Stewart platform.
    The remaining seven columns describe the pose of the moving platform, represented by the position vector and screw parameters:x = [px, py, pz, sx, sy, sz, θ].

3. data/3rrs/data_0110
    This dataset is specifically designed for kinematic modeling and research of 3RRS parallel robots.
    Each sample in the dataset represents a valid motion state of the robot, consisting of three given active joint angles and their corresponding end-effector pose.

    The data is stored in a MATLAB .mat format file.
    The file contains a numerical matrix with a total of 9 columns.

    The first three columns correspond to the joint angles of the three active arms: θ = [theta1, theta2, theta3], serving as the input driving variables for the 3RRS platform.
    The remaining six columns describe the spatial pose of the end-effector, represented by orientation angles (Z-Y-X Euler angles or fixed-axis Euler angles) and position coordinates: X = [phix, phiy, phiz, px, py, pz].
