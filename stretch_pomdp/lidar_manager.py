import rclpy
from rclpy.node import Node
from numpy import linspace, inf
from math import sin, cos
from sensor_msgs.msg import LaserScan

class LidarManager(Node):
    def __init__(self):
        super().__init__('stretch_scan_filter')
        # self.pub = self.create_publisher(LaserScan, '/filtered_scan', 10)
        self.sub = self.create_subscription(LaserScan, '/scan', self.scan_filter_callback, 10)

        self.width = 2
        self.extent = self.width / 2.0
        self.get_logger().info("Subscribing to scan topic, publishing filtered sphere collisions")
        self.spheres = [] # scan points into spheres List(centers)

    def scan_filter_callback(self,msg):
        """
        only read the points that's within 2 meters in x and y coordinates and output them as spheres
        """
        angles = linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        points = [(r * cos(theta), r * sin(theta)) for r,theta in zip(msg.ranges, angles)]
        new_ranges = [r if abs(p[0]) < self.extent and abs(p[1] < self.extent) else inf for r,p in zip(msg.ranges, points)]
        self.spheres = [[r * cos(theta), r * sin(theta), 0] for r, theta in zip(new_ranges, angles) if r != inf]
        print(self.spheres)


def main(args=None):
    rclpy.init(args=args)
    scan_filter = LidarManager()
    rclpy.spin(scan_filter)
    scan_filter.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()