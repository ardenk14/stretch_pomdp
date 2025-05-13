import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
from rclpy.duration import Duration

import sensor_msgs_py.point_cloud2 as pc2

import vamp


class VampManager(Node):
    def __init__(self):
        super().__init__('vamp_manager')

        # Subscribe to Pointcloud and primitives in the environment
        self.sub = self.create_subscription(
            PointCloud2,
            '/vamp_env/point_cloud',
            self.pointcloud_callback,
            10
        )

        # TODO: Add cuboid, capsule, sphere

        # TODO: Subscribe to current configuration

        # TODO: Subscribe to goal poses in the environment

        # TODO: Publish vamp trajectory
        """self.pub = self.create_publisher(
            JointTrajectory,
            '/fake_trajectory',
            10
        )"""

        self.current_config = None
        self.current_trajectory = None
        self.pointcloud = None
        self.spheres = []
        self.capsules = []
        self.cuboids = []
        self.goal_configs = []
        
        (self.vamp_module, self.planner_func, self.plan_settings,
         self.simp_settings) = vamp.configure_robot_and_planner_with_kwargs("stretch", "rrtc")

        sampler = getattr(self.vamp_module, "halton")()
        sampler.skip(0)

    def pointcloud_callback(self, msg):
        pass

    def sphere_callback(self, msg):
        pass

    def capsule_callback(self, msg):
        pass

    def cuboid_callback(self, msg):
        pass

    def current_config_callback(self, msg):
        pass

    def goal_config_callback(self, msg):
        pass

    def check_trajectory(self):
        """ Check if current path leads to collision. Replan if so."""
        # Validate that current pose to next pose is valid
        # If environment has been updated, validate the remainder of the path
        # If either of the above not valid, replan
        pass


def main(args=None):
    rclpy.init(args=args)
    node = VampManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
