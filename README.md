Ros2 package for quick POMDP planning on a Hello-Robot Stretch3

Required packages:
ros2 humble: [here](https://docs.ros.org/en/humble/index.html)
pomdp-py from [here](https://github.com/KavrakiLab/vamp-pomdp/tree/cython) or [here](https://github.com/YC-Liang/Ref-VAMP/tree/main) following installation instructions found [here](https://h2r.github.io/pomdp-py/html/installation.html)
vamp from [here](https://github.com/KavrakiLab/vamp/tree/stretch)

Simulation and trajectory follower: [here](https://github.com/nicholasl23638/stretch_ros2_sim)

Need for experiments:
- [ ] Get trajectory follower working for synchronized base and joints
- [ ] Add actions to POMDP that allow for synchronized base and joints
- [ ] Adjust actions in POMDP to ensure they can well approximate the output from vamp (visualize)
- [ ] Tune micro action size, number of actions to take between replans, and planning time/frequency with real robot trajectory follower

Real-world navigation (desired for experiments):
- [ ] Allow for environment updates
- [ ] Define a number of landmark/goal representations and datastructures for working with each (i.e., base-only or full configuration with some error or gripper with some error etc.)
- [ ] Get observations from another ROS node
- [ ] Get rewards from another ROS node
- [ ] Visualization of environment, landmarks, and goals
- [ ] Visualization of belief state over time and trajectories proposed through returned macro actions

Future Infrastructure:
- [ ] Be able to update landmarks/goals from another ROS node
- [ ] IK to get configurations to plan to for grasping
- [ ] Better vamp planning than just restricting turning angle
- [ ] Visualization of sample vamp plans and pomdp tree
- [ ] Be able to update transition, observation, and reward models + belief representation on the fly through other ROS nodes
