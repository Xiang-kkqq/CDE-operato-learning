# CDE-operato-learning

1. data/data_stewartv13
This dataset is designed for training and validating data-driven forward kinematics models of a 6-6 Stewart platform.
Each sample in the dataset represents a valid configuration of the parallel robot, consisting of six given actuator lengths and the corresponding end-effector pose expressed in screw parameters.

The data are stored in MATLAB .mat format, with files named T1.mat to T8.mat.
Each file contains a variable Tn, which is a numerical matrix with 13 columns.

The first six columns correspond to the actuator lengths of the six legs:L = [l1, l2, l3, l4, l5, l6], representing the input driving variables of the Stewart platform.
The remaining seven columns describe the pose of the moving platform, represented by the position vector and screw parameters:x = [px, py, pz, sx, sy, sz, θ].

2. data/3rrs/data_0110
