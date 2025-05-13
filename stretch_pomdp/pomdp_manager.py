import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
from rclpy.duration import Duration

import sensor_msgs_py.point_cloud2 as pc2

import numpy as np
import sys
print(f"Running with Python: {sys.executable}")
import vamp
import pomdp_py
#from pomdp_py.algorithms.pomcp import POMCP
from stretch_pomdp.problems.stretch.problem import init_stretch_pomdp
from stretch_pomdp.problems.stretch.domain.observation import Observation
from stretch_pomdp.problems.stretch.domain.state import State
from stretch_pomdp.problems.stretch.environments.vamp_template import VAMPEnv
#from pomdp_py.framework.basics import MPPOMDP, sample_explicit_models


class POMDPManager(Node):
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

        # TODO: Subscribe to landmarks and danger zones

        # TODO: Publish vamp trajectory
        """self.pub = self.create_publisher(
            JointTrajectory,
            '/fake_trajectory',
            10
        )"""

        self.current_config = np.array([0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0.])
        self.current_trajectory = None
        self.pointcloud = None
        self.spheres = []
        self.capsules = []
        self.cuboids = []

        self.goal_configs = []
        self.landmarks = []
        self.danger_zones = []

        self.last_action = None

        self.vamp_env = VAMPEnv()
        self.problem = init_stretch_pomdp(self.current_config, self.vamp_env)

        self.planner = pomdp_py.RefPOMDPFast(
                    planning_time=1,
                    exploration_const=0.0,
                    discount_factor=.99,
                    eta=.2,
                    max_depth=25,
                    rollout_depth=50,
                    episode_count=-1,
                    ref_policy_heuristic='entropy',
                    use_prm=False)

        # Create timer for control loop
        self.timer = self.create_timer(1.0, self.control_loop)  # 1Hz control loop

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

    def control_loop(self):
        """ Run the POMDP Planner """
        # Update vamp_env
        if self.last_action is not None:
            # TODO: Get next_state, observation and reward
            observation = Observation(self.current_config)
            state = State(self.current_config, False, False, False)
            # Update history and belief
            self.problem.agent.update_history(self.last_action, observation)
            self.problem.env.apply_transition(state) # current state = next_state (best estimate)
            self.planner.update(self.problem.agent, self.last_action, observation)
        # Plan 
        action = self.planner.plan(self.problem.agent, no_pomdp=False)
        self.last_action = action.action_sequence[0]

        # TODO: Publish action
        self.get_logger().info(str(action))
        
        """# Sample explicit models -> Uses model to get true next state given action
        next_state, observation, reward, nsteps = sample_explicit_models(T=self.problem.env.transition_model,
                                                    O=self.problem.agent.observation_model,
                                                    R=self.problem.env.reward_model,
                                                    state=self.problem.env.state,
                                                    action=action,
                                                    discount_factor=self.planner.discount_factor)
        self.problem.agent.update_history(action, observation)
        # Update history and belief for next run
        self.problem.env.apply_transition(next_state) # Current state = next_state
        self.planner.update(self.problem.agent, action, observation)"""

def main(args=None):
    rclpy.init(args=args)
    node = POMDPManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
