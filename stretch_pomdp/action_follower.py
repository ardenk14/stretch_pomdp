import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

import numpy as np
import math
from collections import deque
import copy

from rclpy.time import Time
import matplotlib.pyplot as plt
import rerun as rr

def get_next_position(position, action):
    """
    Transition function for the navigation model.

    :param position: agent current position (x, y, yaw, ...) in world frame.
    :param action: action in the robot's frame (dx, dy, dyaw, ...)
                where dx is forward, dy is left, and dyaw is rotation.
    :return: the next position in the world frame.
    """
    next_position = np.zeros_like(position)
    x, y, yaw = position[0], position[1], position[2]

    dx_body = action[0]
    dy_body = action[1]
    dyaw = action[2]

    dx_world = dx_body * np.cos(yaw) - dy_body * np.sin(yaw)
    dy_world = dx_body * np.sin(yaw) + dy_body * np.cos(yaw)

    next_position[0] = x + dx_world
    next_position[1] = y + dy_world
    next_position[2] = yaw + dyaw
    next_position[3:] = position[3:] + action[3:]

    return next_position

def get_action_from_positions(initial_position, final_position):
    """
    Compute the local-frame action that moves the robot from initial_position to final_position.

    :param initial_position: (x, y, yaw, ...) in world frame
    :param final_position: (x, y, yaw, ...) in world frame
    :return: action (dx_body, dy_body, dyaw, ...) in the robot's local frame at the initial position
    """
    x0, y0, yaw0 = initial_position[0], initial_position[1], initial_position[2]
    x1, y1, yaw1 = final_position[0], final_position[1], final_position[2]

    # Position difference in world frame
    dx_world = x1 - x0
    dy_world = y1 - y0
    dyaw = yaw1 - yaw0

    # Rotate the world-frame delta into the robot's body frame at yaw0
    dx_body = dx_world * np.cos(yaw0) + dy_world * np.sin(yaw0)
    dy_body = -dx_world * np.sin(yaw0) + dy_world * np.cos(yaw0)

    # Include any other state dimensions
    action = np.zeros_like(initial_position)
    action[0] = dx_body
    action[1] = dy_body
    action[2] = dyaw
    action[3:] = final_position[3:] - initial_position[3:]

    return action

class ActionFollower(Node):
    def __init__(self):
        super().__init__('action_follower')

        #rr.init("rerun_example_my_data", spawn=False)  # Don't spawn a new viewer
        #rr.connect_grpc()

        # TODO: What is the timing map between action.py and here?
        self.T = 0.7#0.2
        self.V = 0.07#0.3#0.1 Really 0.055

        self.current_config = np.array([0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0.])
        self.last_config = np.array([0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0.])
        self.actions = deque()
        self.last_action = None
        self.last_pos = np.array([0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0.])

        self.total = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.n = 0

        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd = cmd

        self.vs = []
        self.ws = []
        self.ts = []
        self.start_t = None

        # Subscribe to actions and odometry
        self.actions_sub = self.create_subscription(
            Float64MultiArray, 'actions', self.action_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)
        
        # Publish the twist and joint pose command
        self.cmd_vel_pub = self.create_publisher(Twist, '/stretch/cmd_vel', 10)
        self.joint_point_pub = self.create_publisher(Float64MultiArray, "/joint_pose_cmd", 10)
        self.act_obs_pub = self.create_publisher(Float64MultiArray, '/act_obs', 1)

        # Set timer to run each macro action for n seconds and grab the observation at the end
        self.control = self.create_timer(0.1, self.control_loop) # Time must match the time parameter in the POMDP actions.py
        self.action_update = self.create_timer(self.T, self.update_action)

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

        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        t = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9

        self.vs.append(v)
        self.ws.append(w)
        if self.start_t is not None:
            self.ts.append(t - self.start_t)
        else:
            self.ts.append(0.0)
            self.start_t = t

        #print("LIN VEL: ", v)
        #print("ANG VEL: ", w)
        #print("T: ", t)

    def action_callback(self, msg):
        self.actions = deque(np.array(msg.data).reshape([dim.size for dim in msg.layout.dim]))
        #self.actions.append(self.actions[-1])

    def control_loop(self):
        if len(self.ts) > 0 and self.ts[-1] < 3.0:
            cmd = Twist()
            cmd.linear.x = 0.3
            cmd.angular.z = 0.5
        else:
            cmd = Twist()
            cmd.linear.x = 0.3
            cmd.angular.z = -0.5
        self.cmd_vel_pub.publish(cmd)
        #self.cmd_vel_pub.publish(self.cmd)

    def update_action(self):        
        if self.last_action is not None:
            #action = get_action_from_positions(self.last_config, obs)
            #tru = "TRUE ACTION: " + str(action[:3])
            #self.get_logger().warning(tru)
            #self.last_config = copy.deepcopy(obs)
            #hyp = "ACTION: " + str(self.last_action[:3])
            #self.get_logger().warning(hyp)
            msg = Float64MultiArray()

            action = list(self.last_action)
            obs = list(self.current_config)
            data = action + obs

            # You could encode size info in dim labels or encode them separately elsewhere
            dim = MultiArrayDimension()
            dim.label = "flat"   # single dimension
            dim.size = len(data)
            dim.stride = len(data)

            msg.layout.dim = [dim]
            msg.layout.data_offset = 0
            msg.data = data

            self.act_obs_pub.publish(msg)

            """msg = Float64MultiArray()

            dim1 = MultiArrayDimension()
            dim1.label = "rows"
            dim1.size = 2
            dim1.stride = len(self.last_action) # rows * cols

            dim2 = MultiArrayDimension()
            dim2.label = "cols"
            dim2.size = len(self.last_action)
            dim2.stride = len(self.last_action) # cols

            msg.layout.dim = [dim1, dim2]
            msg.layout.data_offset = 0

            data = list(self.last_action)
            obs = self.current_config
            data.extend(obs)
            msg.data = data

            self.act_obs_pub.publish(msg)"""

        if len(self.actions) < 1:
            #if len(self.actions) == 1:
            #    self.last_action = self.actions.popleft()
            #else:
            #    self.last_action = None
            self.last_action = None
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd = cmd
            return
        
        # Set new action
        action = self.actions.popleft()
        self.last_action = action

        # Set and publish the twist for this action
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        if action[0] != 0.0:
            cmd.linear.x = action[0]#self.V if action[0] > 0.0 else -self.V
            cmd.angular.z = action[1]#action[2] / self.T #(1/0.045)
        self.cmd = cmd

        # TODO: Set and publish the joint commands for this action

def main(args=None):
    rclpy.init(args=args)
    
    # Create the action follower node
    follower_node = ActionFollower()
    
    try:
        rclpy.spin(follower_node)
    except KeyboardInterrupt:
        pass
    finally:
        plt.scatter(follower_node.ts, follower_node.ws)
        plt.scatter(follower_node.ts, follower_node.vs)
        plt.show()
        follower_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()