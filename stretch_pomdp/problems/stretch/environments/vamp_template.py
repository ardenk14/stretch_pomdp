"""
This code sets up a template environment for integrating with VAMP.
"""

import vamp 
import numpy as np
import math
from pomdp_py.utils.transformations import aabb_collision_check
from vamp import pybullet_interface as vpb
from pathlib import Path
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
                 robot_init_config=(-0.9, -1.4, 0., 0.5, 0., 0., 0., 0., 0., 0., 0., 0, 0),
                 lab_map_dir = None,
                 obstacle_loc = None,
                 debug=False,
                 resize=0):
        vamp.sphere.set_radius(0.3)
        # 2d homogeneous transformation matrix
        self.robot_to_world = np.array([-0.9, -1.4])
        self.robot_yaw_to_world_yaw = math.pi / 2
        self.rPw = np.array([[0, -1, -0.9],
                             [1,  0, -1.4],
                             [0,  0,  1]])
        self._robot_init_config = robot_init_config

        # Customize objects.
        # ====================================================
        # Assumes a single spherical goal region.
        self._goal = (1.0, [-0.9, 1.3, 0.0, 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.sphere_approx_radius = 0.2
        # self.cylinder_height = 1
        # self.cylinder_euler = (0, 0, 1.16)

        self._landmarks = [[-0.48, 0, 0,],
                           [-1.43, 0, 0,],
                           [0.48, 0, 0,],
                           [1.43, 0, 0],
                           [-0.48, 1, 0,],
                           [-1.43, 1, 0,],
                           [-0.48, -1, 0,],
                           [-1.43, -1, 0,],]
        
        # experiment depth image
        self.depth_img = np.zeros((4, 4))
        self.depth_img[1:3, 1:3] = 255
        # self.img_to_heightfield(self.depth_img)

        self.lab_map_path = lab_map_dir
        self.lab_width = 380
        self.lab_height = 400
        self.scale_x = 3.8
        self.scale_y = 4

        #self._landmarks = [
        #    [[1, 1, 1], [2.0, 1.0, 0.0], [0, 0, 0]],
        #    [[1, 1, 1], [2.0, -1.0, 0.0], [0, 0, 0]],
        #    [[1, 1, 1], [1.0, 1.5, 0.0], [0, 0, 0]],
        #    [[1, 1, 1], [1.0, -1.5, 0.0], [0, 0, 0]],
        #    [[1, 1, 1], [-0.5, 0.0, 0.0], [0, 0, 0]],
        #]

        self._danger_zones = [
            # [[7, 1, 1], [6.0, -9.0, 4.0], [0, 0, 0]],
            # [[7, 1, 1], [6.0, -13.0, 4.0], [0, 0, 0]],
            # [[5, 0.8, 1], [6.0, 21.0, 4.0], [0, 0, 0]],
            # [[0.5, 2, 1], [-22.5, 1.0, 4.0], [0, 0, 0]],
        ]

        self.cuboids = [[[0.9, 0.6, 0.5], [1, -1.4, 0.5], [0, 0, 0]],
                        [[0.5, 0.6, 0.5], [1.3, 1.25, 0.5], [0, 0, 0]]]
        self.spheres = [] # (self.sphere_approx_radius, (1.5, 0, 0))
        self.heightfields = []
        self.cylinders = []
        self.capsules = []

        self.env = vamp.Environment()
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
        # load dynamic obstacles in the state
        env.add_sphere(vamp.Sphere(list(state.get_obstacle_loc) + [0.],
                                   self.sphere_approx_radius))
        
        # load static obstacles (only boxes atm)
        for half_extents, center, euler in self.cuboids:
            self.env.add_cuboid(vamp.Cuboid(center, euler, half_extents))
        return env

    def init_env(self):
        self.load_primitive_collision_objects()

    def sample_landmark_zones(self, lm_index):
        raise NotImplementedError

    def visualize_key_features(self):
        """Customize to add key objects (e.g. danger zones, landmarks, goal, starting regions) to the GUI."""
        # rr.log("landmarks", rr.Points3D(self._landmarks, radii=0.1))
        centers = []
        radii = []
        for radius, center in self.spheres:
            centers.append(center)
            radii.append(radius)
        rr.log("Obstacles", rr.Points3D(centers, radii=radii), static=True)
        rr.log("Goal", rr.Points3D(self._goal[1][:3], radii=0.1), static=True)
        # rr.log("Start", rr.Points3D(self.get_robot_init_config[:3], radii=0.1), static=True)

        for i, (half_extents, center, euler) in enumerate(self.cuboids):
            rr.log(["box", str(i)], rr.Boxes3D(half_sizes=half_extents, centers=center, fill_mode="solid", colors=[125,125,125]))

        rr.log("box/plane", rr.Boxes3D(sizes=(3.8, 4., 0.1), centers=(0, 0, -0.05)))
        return

    def img_to_heightfield(self, img):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                size = (1, 1, (255 - img[i, j]) / 255)
                center = (i, j, 0.5)
                if size[2] > 0:
                    rr.log(["wall", str(i), str(j)], rr.Boxes3D(sizes=size, centers=center, colors=(125,125,125)))
        return

    def get_landmarks_pos(self, include_goal=True):
        lms = [lm + [0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0] for lm in self._landmarks]
        if include_goal:
            lms.append(self._goal[1])
        return lms

    def load_primitive_collision_objects(self):
        for half_extents, center, euler in self.cuboids:
            self.env.add_cuboid(vamp.Cuboid(center, euler, half_extents))
        
        if self.lab_map_path is not None:
            print("loading a heightmap")
            hf = vamp.png_to_heightfield(
                self.lab_map_path,
                (0, 0, 0.),
                (1./self.lab_width * self.scale_x, 1./self.lab_height * self.scale_y, 1)
            )
            self.env.add_heightfield(hf)
        return

        # for radius, center in self.spheres:
        #     self._env.add_sphere(vamp.Sphere(center, radius))

        

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
        vamp_env = self.env if vamp_env is None else vamp_env
        colliding_jnts = vamp.stretch.sphere_validity(config, vamp_env)
        for j in colliding_jnts:
            if len(j):
                return True
        return False
    
    def collision_validate(self, config, vamp_env = None):
        vamp_env = self.env if vamp_env is None else vamp_env
        return not vamp.stretch.validate(config, vamp_env)
    
    def has_collision_sphere(self, config, vamp_env = None):
        vamp_env = self.env if vamp_env is None else vamp_env
        colliding_jnts = vamp.sphere.sphere_validity(config[:3], vamp_env)
        for j in colliding_jnts:
            if len(j):
                return True
        return False

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
    
    def load_pb_visualiser(self):
        from vamp import pybullet_interface as vpb
        self.pb = vpb.PyBulletSimulator("", [], True)
        self.sim = self.pb.client
        self.pb.client.configureDebugVisualizer(self.pb.client.COV_ENABLE_GUI, 1)
        for half_extents, center, euler in self.cuboids:
            quat = quaternion_from_euler(*euler)
            self.pb.add_cuboid(half_extents, center, quat)


def create_robot_params_gui(vmp_env):
    """Create debug params to set the robot positions from the GUI."""
    params = {}
    for name in ["x", "y"]:
        params[name] = vmp_env.sim.addUserDebugParameter(
            name,
            rangeMin=-40,
            rangeMax=40,
            startValue=0,
            physicsClientId=vmp_env.sim._client,
        )
    params["z"] = vmp_env.sim.addUserDebugParameter(
        "z",
        rangeMin=-10,
        rangeMax=10,
        startValue=0,
        physicsClientId=vmp_env.sim._client,
    )
    for name in ["roll", "pitch", "yaw"]:
        params[name] = vmp_env.sim.addUserDebugParameter(
            name,
            rangeMin=-np.pi,
            rangeMax=np.pi,
            startValue=0.,
            physicsClientId=vmp_env.sim._client,
        )
    return params


def read_robot_params_gui(robot_params_gui, vmp_env):
    """Read robot configuration from the GUI."""
    return np.array(
        [
            vmp_env.sim.readUserDebugParameter(
                param,
                # physicsClientId=vmp_env._sim._client,
            )
            for param in robot_params_gui.values()
        ]
    )

def main():
    env_gui = VAMPEnv()
    env_gui.load_pb_visualiser()
    env_gui.pb.add_sphere(0.3,  env_gui.get_robot_init_config[:3], None, "blue")
    idx = env_gui.pb.add_sphere(0.3,  env_gui.get_robot_init_config[:3], None, "red")
    robot_params_gui = create_robot_params_gui(env_gui)

    while True:
        config = read_robot_params_gui(robot_params_gui, env_gui)
        env_gui.pb.update_object_position(idx, config[:3])
        robot_config = list(config[:3]) + [0.5, 0., 0., 0., 0., 0., 0., 0., 0, 0]
        collision = env_gui.has_collision_sphere(robot_config)
        print(f"collision: {collision}")

if __name__ == "__main__":
    main()