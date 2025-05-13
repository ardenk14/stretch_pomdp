from pomdp_py.framework import basics
import numpy as np


class State(basics.State):
    EPSILON = 0.1
    UPPER_BOUNDS = np.array([5, 5, 2*np.pi, 1.1, 0.13, 0.13, 0.13, 0.13, 0, 0, 0, 0, 0])#np.array([5, 5, 2*np.pi, 0, 0, 0, 0, 0, 0, 0, 0])
    LOWER_BOUNDS = np.array([-5, -5, -2*np.pi, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def __init__(self, position, danger_zone, landmark, goal):
        """
        position (numpy array): The state position (x, y, theta, lift, extenstion, wrist_yaw, wrist_pitch, wrist_roll, head_pan, head_tilt, gripper)).
        danger_zone (bool): The robot is at a danger zone.
        landmark (bool): The robot is at a landmark.
        goal (bool): The robot is at the goal.

        """
        self._position = position
        self._terminal = danger_zone or goal
        self._danger_zone = danger_zone
        self._landmark = landmark
        self._goal = goal
        self._hash = hash((tuple(self._position), self._terminal, self._danger_zone, self._landmark, self._goal))

    @property
    def get_position(self):
        return self._position

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if isinstance(other, State):
            return (np.linalg.norm(np.array(self._position) - np.array(other._position), ord=3) < State.EPSILON and
                    tuple(self._position) == tuple(other._position) and
                    self._terminal == other._terminal and
                    self._danger_zone == other._danger_zone and
                    self._landmark == other._landmark and
                    self._goal == other._goal)
        else:
            return False

    def __str__(self):
        return f"<pos: {self._position} | dz: {self._danger_zone} | lm: {self._landmark} | goal: {self._goal}>"

    def __repr__(self):
        return self.__str__()

    @property
    def terminal(self):
        return self._terminal

    @property
    def is_goal(self):
        return self._goal