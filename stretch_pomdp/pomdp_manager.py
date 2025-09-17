import rclpy
import tf2_ros
from rclpy.node import Node
from scalene.scalene_utility import on_stack

from sensor_msgs.msg import PointCloud2
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
from rclpy.duration import Duration
import threading
from rclpy.time import Time

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist, PoseStamped, Point

import sensor_msgs_py.point_cloud2 as pc2

import numpy as np
import math
import sys
print(f"dRunning with Python: {sys.executable}")
import vamp
import pomdp_py
#from pomdp_py.algorithms.pomcp import POMCP
from stretch_pomdp.problems.stretch.problem import init_stretch_pomdp
from stretch_pomdp.problems.stretch.domain.observation import Observation
from stretch_pomdp.problems.stretch.domain.action import Action
from stretch_pomdp.problems.stretch.domain.state import State
from stretch_pomdp.problems.stretch.domain.action import MacroAction
from stretch_pomdp.problems.stretch.domain.transition_model import StretchTransitionModel
from stretch_pomdp.problems.stretch.environments.vamp_template import VAMPEnv
#from pomdp_py.framework.basics import MPPOMDP, sample_explicit_models
from pomdp_py.framework.basics import MacroObservation
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import MultiArrayDimension, MultiArrayLayout

import rerun as rr
from collections import deque
import copy
import tf2_ros
from tf2_ros import TransformException
import time


class POMDPManager(Node):
    def __init__(self):
        super().__init__('vamp_manager')

        rr.init("rerun_example_my_data", spawn=True)
        #rr.serve_web(open_browser=True, server_memory_limit='25.00%')
        #rr.serve_web(server_memory_limit='0.00%')

        # Subscribe to Pointcloud and primitives in the environment
        self.sub = self.create_subscription(
            PointCloud2,
            '/vamp_env/point_cloud',
            self.pointcloud_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)
        
        self.act_obs_sub = self.create_subscription(Float64MultiArray, '/act_obs', self.act_obs_callback, 1)

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
        #self.trajectory_pub = self.create_publisher(Path, 'trajectory', 10)
        self.action_pub = self.create_publisher(Float64MultiArray, 'actions', 1)

        self.lock = threading.Lock()

        # get initial environment sensing
        # self.static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.obstacle_loc = None # (1, 0, 0)

        self.current_config = np.array([0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.obs_lst = []
        self.acts_lst = []
        self.current_trajectory = None
        self.pointcloud = None
        self.spheres = []
        self.capsules = []
        self.cuboids = []

        self.goal_configs = []
        self.landmarks = []
        self.danger_zones = []

        self.last_action = None
        self.last_v = 0.0
        self.last_w = 0.0
        self.last_t = None

        self.vamp_env = VAMPEnv(obstacle_loc=self.obstacle_loc)
        self._Tm = StretchTransitionModel(self.vamp_env)
        self.problem = None

        self.planner = pomdp_py.ROPRAS3(
            planning_time=2,
            max_depth=20,
            rollout_depth=20,
            eta=0.2,
            ref_policy_heuristic='uniform',
            use_prm=False
        )

        # Create timer for control loop
        self.aruco_timer = self.create_timer(1, self._get_obstacle_transform)
        self.detect_loop()
        self.control_timer = self.create_timer(10.0, self.control_loop)
        self.control_loop()

    def _get_obstacle_transform(self):
        try:
            # Look up transform between the base_link and requested ArUco tag
            transform = self.tf_buffer.lookup_transform('base_link',
                                                        'cube',
                                                        Time())

            self.obstacle_loc = [transform.transform.translation.x,
                                 transform.transform.translation.y,
                                 0] # we use cylinder to approximate anything, hence we assume things are connected to the ground
        except TransformException as ex:
            pass

    def odom_callback(self, msg):
        def quaternion_to_yaw(quaternion):
            """Convert quaternion to yaw angle (rotation around z-axis)"""
            # Extract the yaw angle from the quaternion
            # This is a simplified conversion assuming the robot moves in a plane
            siny_cosp = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
            cosy_cosp = 1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            return yaw
        current_pose = msg.pose.pose
        self.current_config[0] = current_pose.position.x
        self.current_config[1] = current_pose.position.y
        self.current_config[2] = quaternion_to_yaw(current_pose.orientation)
        rr.log("Current_Pos", rr.Arrows3D(origins=list(self.current_config[:2])+[0.0], vectors=[0., 0., 1.], colors=[0.2, 1.0, 0.2]))

        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        t = Time.from_msg(msg.header.stamp)

        #print(msg.header.stamp)
        if self.last_t is not None:
            diff = (t - self.last_t).nanoseconds * 1e-9
            #print("LINEAR ACCEL: ", (v - self.last_v) / diff)
            #print("ANGULAR ACCEL: ",(w - self.last_w) / diff)

        self.last_v = v
        self.last_w = w
        self.last_t = t

    def act_obs_callback(self, msg):
        flat_data = msg.data
        act_len = len(Action("None")._motion)
        action = flat_data[:act_len]
        observation = flat_data[act_len:]
        act = Action("None")
        for k, v in act.MOTIONS.items():
            if list(v) == list(action):
                act = Action(k)
        print(act)
        #if act._name != "None":
        self.lock.acquire()
        self.obs_lst.append(Observation(observation))
        self.acts_lst.append(act)
        self.lock.release()
        rr.log("Observation", rr.Arrows3D(origins=list(observation[:2])+[0.0], vectors=[0., 0., 1.], colors=[0.2, 1.0, 1.0]))

    def pointcloud_callback(self, msg):
        # Update vamp_env
        #print(problem.policty.root)
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

    def yaw_to_quaternion(self, yaw):
        """Convert yaw angle to quaternion"""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(0.0)
        sp = math.sin(0.0)
        cr = math.cos(0.0)
        sr = math.sin(0.0)
        
        qw = cy * cp * cr + sy * sp * sr
        qx = cy * cp * sr - sy * sp * cr
        qy = cy * sp * cr + sy * cp * sr
        qz = sy * cp * cr - cy * sp * sr
        # Update vamp_env
        #print(problem.policty.root)
        
        return qx, qy, qz, qw

    def control_loop(self):
        thread = threading.Thread(target=self._run_control_loop)
        thread.daemon = True
        thread.start()

    def detect_loop(self):
        thread = threading.Thread(target=self._get_obstacle_transform)
        thread.daemon = True
        thread.start()

    def _run_control_loop(self):
        """ Run the POMDP Planner """
        if self.problem is None:
            print("WAITING FOR ENVIRONMENT INITIALIZATION")
            if self.obstacle_loc is not None:
                print(f"obstacle_loc is: {self.obstacle_loc}")
                self.problem = init_stretch_pomdp(self.current_config, self.obstacle_loc, self.vamp_env)
            else:
                return

        print("BEGIN CONTROL LOOP")
        # Update vamp_env
        #print(problem.policty.root)

        self.lock.acquire()
        action = copy.deepcopy(self.acts_lst)
        observation = copy.deepcopy(self.obs_lst)
        self.acts_lst = []
        self.obs_lst = []
        self.lock.release()

        if len(action) != 0:
            # TODO: Get next_state, observation and reward
            observation = MacroObservation(observation)
            action = MacroAction(action)
            print("ACT: ", action)
            print("OBS: ", observation)
            state = State(self.current_config, False, False, False)

            # Update history and belief
            self.problem.agent.update_history(action, observation)
            self.problem.env.apply_transition(state) # current state = next_state (best estimate)
            belief_start_time = time.time()
            self.planner.update(self.problem.agent, action, observation)
            belief_time = time.time() - belief_start_time
            print(f"Belief update time: {belief_time}")

            # Visualize current belief
            positions = []
            obstacle_positions = []
            for k, v in self.problem.agent.tree.belief.get_histogram().items():
                positions.append(list(k.get_position[:2]) + [0.0])
                print(f"belief particle obstacle loc: {k.get_obstacle_loc}")
                obstacle_positions.append(list(k.get_obstacle_loc))
                
            rr.log("current_pos_belief", rr.Points3D(np.array(positions)))
            # cylinder_radius = [self.problem._vamp_env.cylinder_approx_radius] * len(obstacle_positions)
            # cylinder_length = [self.problem._vamp_env.cylinder_height] * len(obstacle_positions)
            # cylinder_euler = [self.problem._vamp_env.cylinder_euler] * len(obstacle_positions)
            # rr.log("current_obs_belief", rr.Capsules3D(lenghts = cylinder_length, 
            #                                            radii = cylinder_radius, 
            #                                            translations=obstacle_positions, 
            #                                            rotation_axis_angles=cylinder_euler,
            #                                            colors = [0, 1, 0,]))

        # get next action
        t1 = time.time()
        action = self.planner.plan(self.problem.agent, no_pomdp=False)
        t2 = time.time()
        print("PLANNING TIME: ", t2-t1)
        reference_policy_time = self.problem.agent.policy_model.total_time
        vamp_time = self.problem.agent.policy_model.rrtc_time
        print(f"reference_policy_time: {reference_policy_time}")
        print(f"vamp time: {vamp_time}")
        self.problem.agent.policy_model.total_time = 0.0
        self.problem.agent.policy_model.rrtc_time = 0.0
        self.last_action = action.action_sequence[0]
        self.get_logger().info(str(action))#.action_sequence[0].motion)

        # Publish actions
        act_lst = []
        for act in action.action_sequence:
            act_lst.extend(act.motion)
        
        msg = Float64MultiArray()

        dim1 = MultiArrayDimension()
        dim1.label = "rows"
        dim1.size = len(action.action_sequence)
        dim1.stride = len(action.action_sequence) * len(action.action_sequence[0].motion) # rows * cols

        dim2 = MultiArrayDimension()
        dim2.label = "cols"
        dim2.size = len(action.action_sequence[0].motion)
        dim2.stride = len(action.action_sequence[0].motion) # cols

        msg.layout.dim = [dim1, dim2]
        msg.layout.data_offset = 0

        msg.data = [float(i) for i in act_lst]

        self.action_pub.publish(msg)

        # VISUALIZATIONS ------------------------------------------------------------------------------------

        # Visualize obstacles
        self.vamp_env.visualize_key_features()

        # Visualize current belief
        #positions = []
        #for k, v in self.problem.agent.tree.belief.get_histogram().items():
        #    positions.append(list(k.get_position[:2]) + [0.0])
        #rr.log("current_belief", rr.Points3D(np.array(positions)))

        # visualize tree
        q = deque()
        q.append(self.problem.agent.tree)
        origins = []
        vectors = []
        colors = []
        beliefs = []
        size = 1
        while len(q) != 0:
            node = q.popleft()
            size += 1
            #print("VISITED NODE: ", node)
            if len(node.belief.particles) != 0:
                avg = sum([s._position for s in node.belief.particles]) / len(node.belief.particles)
                beliefs.extend([list(s._position)[:2] + [0.0] for s in node.belief.particles])
                #print("AVG: ", avg)
            children = list(node.children.values())
            for child in children: #Qnode
                q.extend(list(child.children.values()))
                for new_node in list(child.children.values()):
                    if len(new_node.belief.particles) != 0:
                        new_avg = sum([s._position for s in new_node.belief.particles]) / len(new_node.belief.particles)
                        origin = avg[:3]
                        origin[2] = 0.0
                        origins.append(origin)
                        vector = new_avg[:3]
                        vector[2] = 0.0
                        vectors.append(vector-origin)
                        color = [1.0, 0.0, 0.0] if new_node.V < 0 else [0.0, 1.0, 0.0]
                        colors.append(color)

        print("TREE SIZE: ", size)
        #rr.log("Tree", rr.Clear(recursive=False))
        print(f"number of edges in the tree: {vectors}")
        rr.log("Tree", rr.Arrows3D(origins=origins, vectors=vectors, colors=colors))
        rr.log("Beliefs", rr.Points3D(beliefs))

        # Visualize actions to be taken
        position = self.current_config
        origins = []
        vectors = []
        for act in action.action_sequence:
            #act = act._motion
            next_position = self._Tm.get_next_position(position, act)
            origins.append(position[:3])
            vectors.append(next_position[:3] - position[:3])
            origins[-1][-1] = 0.05
            vectors[-1][-1] = 0.0
            position = next_position
        rr.log("Taken_Action", rr.Arrows3D(origins=origins, vectors=vectors, colors=[0.0, 0.5, 1.0]))
        print("END CONTROL LOOP")


def main(args=None):
    print("REACHED POMDP MANAGER")
    rclpy.init(args=args)
    node = POMDPManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

