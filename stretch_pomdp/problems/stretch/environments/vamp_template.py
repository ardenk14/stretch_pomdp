"""
This code sets up a template environment for integrating with VAMP.
"""

import vamp 
import numpy as np
import math
from pomdp_py.utils.transformations import aabb_collision_check

import rerun as rr

def quaternion_from_euler(r, p, y):
    rs2 = math.sin(r / 2)
    rc2 = math.cos(r / 2)
    ps2 = math.sin(p / 2)
    pc2 = math.cos(p / 2)
    ys2 = math.sin(y / 2)
    yc2 = math.cos(y / 2)

    qx = rs2 * pc2 * yc2 - rc2 * ps2 * ys2
    qy = rc2 * ps2 * yc2 + rs2 * pc2 * ys2
    qz = rc2 * pc2 * ys2 - rs2 * ps2 * yc2
    qw = rc2 * pc2 * yc2 + rs2 * ps2 * ys2

    return [qw, qx, qy, qz]


class VAMPEnv():

    def __init__(self,
                 robot_init_config=(0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.),
                 obstacle_loc = None,
                 debug=False,
                 resize=0):
        self._robot_init_config = robot_init_config

        # Customize objects.
        # ====================================================
        # Assumes a single spherical goal region.
        self._goal = (1.0, [2.5, 0.0, 0.0, 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.sphere_approx_radius = 0.5
        # self.cylinder_height = 1
        # self.cylinder_euler = (0, 0, 1.16)

        self._landmarks = []
        for i in range(-1, 4):
            for j in range(-1, 4):
                self._landmarks.append([[1, 1, 1], [float(i)/2.0, float(j)/2.0, 0], [0, 0, 0]])
        #self._landmarks = [
        #    [[1, 1, 1], [2.0, 1.0, 0.0], [0, 0, 0]],
        #    [[1, 1, 1], [2.0, -1.0, 0.0], [0, 0, 0]],
        #    [[1, 1, 1], [1.0, 1.5, 0.0], [0, 0, 0]],
        #    [[1, 1, 1], [1.0, -1.5, 0.0], [0, 0, 0]],
        #    [[1, 1, 1], [-0.5, 0.0, 0.0], [0, 0, 0]],
        #]

        self._danger_zones = [
            [[7, 1, 1], [6.0, -9.0, 4.0], [0, 0, 0]],
            [[7, 1, 1], [6.0, -13.0, 4.0], [0, 0, 0]],
            [[5, 0.8, 1], [6.0, 21.0, 4.0], [0, 0, 0]],
            [[0.5, 2, 1], [-22.5, 1.0, 4.0], [0, 0, 0]],
        ]

        self.cuboids = []
        self.spheres = []
        self.heightfields = []
        self.cylinders = []
        self.capsules = []
        # ====================================================

        self._env = vamp.Environment()
        self.init_env()

    @property
    def get_goal_pos(self):
        return self._goal[1]

    @property
    def get_num_lms(self):
        return len(self._landmarks)

    @property
    def get_robot_init_config(self):
        return self._robot_init_config
    
    def state_to_vamp(self, state):
        """
        Pass in a state, load the state obstacles into vamp and return the 
        new vamp environment
        """
        env = vamp.Environment()
        env.add_sphere(vamp.Sphere(state.get_obstacle_loc,
                                   self.sphere_approx_radius))
        return env

    def init_env(self):
        self.load_primitive_collision_objects()

    def sample_landmark_zones(self, lm_index):
        raise NotImplementedError

    def visualize_key_features(self):
        """Customize to add key objects (e.g. danger zones, landmarks, goal, starting regions) to the GUI."""
        centers = []
        radii = []
        for radius, center in self.spheres:
            centers.append(center)
            radii.append(radius)
        rr.log("Obstacles", rr.Points3D(centers, radii=radii), static=True)
        rr.log("Goal", rr.Points3D(self._goal[1][:3], radii=0.1), static=True)

    def get_landmarks_pos(self, include_goal=True):
        lms = [lm[1] + [0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0] for lm in self._landmarks]
        if include_goal:
            lms.append(self._goal[1])
        return lms

    def load_primitive_collision_objects(self):
        for radius, center in self.spheres:
            self._env.add_sphere(vamp.Sphere(center, radius))

        # print("CENTER: ", center)
        # print("RADIUS: ", radius)
        # rr.log("Obstacles", rr.Points3D(center, radii=[radius]), static=True)

        """for radius, height, center, euler in self.capsules:
            self._env.add_capsule(vamp.make_cylinder(radius, height, *center, *euler))

        for half_extents, center, euler in self.cuboids:
            self._env.add_cuboid(vamp.make_cuboid(*center, *euler, *half_extents))

        for path, center, scale in self.heightfields:
            scale_temp = list(scale)
            scale_temp[2] = (1. / scale_temp[2])
            center_temp = list(center)
            center_temp[2] = center_temp[2] + (scale_temp[2] / 2)

            self._env.add_heightfield(vamp.png_to_heightfield(path, center, scale))"""

    def collision_checker(self, config, vamp_env = None):
        """ Returns True if in collision, False otherwise."""
        # Upper bound 0.13
        # vamp_config = [config[0], config[1], config[2], config[3]]
        vamp_env = self._env if vamp_env is None else vamp_env
        return not vamp.stretch.validate(config, vamp_env)

    def dz_checker(self, config):
        # work for axis aligned danger zone bounding box only
        for sphere in self.spheres:
            if np.linalg.norm(np.array(config[0:2]) - np.array(sphere[1])[:2]) < sphere[0]:
                return True
        #for half_extent, center, _ in self._danger_zones:
        #    if aabb_collision_check(half_extent, center, config[:3]):
        #        return True
        return False

    def lm_checker(self, config):
        # work for axis aligned landmark boxes only
        for half_extent, center, _ in self._landmarks:
            if aabb_collision_check(half_extent, center, config[:3]):
                return True
        return False

    def goal_checker(self, config):
        return np.linalg.norm(np.array(config[0:2]) - np.array(self._goal[1][:2])) < 0.1#self._goal[0]
