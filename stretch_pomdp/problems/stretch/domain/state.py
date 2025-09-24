from pomdp_py.framework import basics
import numpy as np


class State(basics.State):
    EPSILON = 0.01
    UPPER_BOUNDS = np.array([5, 5, 2*np.pi, 1.1, 0.13, 0.13, 0.13, 0.13, 0, 0, 0, 0, 0]) # np.array([5, 5, 2*np.pi, 0, 0, 0, 0, 0, 0, 0, 0])
    LOWER_BOUNDS = np.array([-5, -5, -2*np.pi, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def __init__(self, position, obstacle_loc, danger_zone, goal):
        """
        position (numpy array): The state position (x, y, theta, lift, extenstion, wrist_yaw, wrist_pitch, wrist_roll, head_pan, head_tilt, gripper)).
        danger_zone (bool): The robot is at a danger zone.
        landmark (bool): The robot is at a landmark.
        goal (bool): The robot is at the goal.

        """
        self._position = position
        self._obstacle_loc = obstacle_loc
        self._terminal = danger_zone or goal
        self._danger_zone = danger_zone
        # self._landmark = landmark
        self._goal = goal
        self._hash = hash((tuple(self._position), tuple(self._obstacle_loc)))

    @property
    def get_position(self):
        return self._position
    
    @property
    def get_obstacle_loc(self):
        return self._obstacle_loc

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if isinstance(other, State):
            return (np.linalg.norm(np.array(self._position) - np.array(other._position), ord=3) < State.EPSILON and 
                    np.linalg.norm(np.array(self._obstacle_loc) - np.array(other._obstacle_loc), ord=3) < State.EPSILON)
        else:
            return False

    def __str__(self):
        return f"<pos: {self._position} | dz: {self._danger_zone} | goal: {self._goal}>"

    def __repr__(self):
        return self.__str__()

    @property
    def terminal(self):
        return self._terminal

    @property
    def is_goal(self):
        return self._goal