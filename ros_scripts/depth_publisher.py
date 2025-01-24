#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
import cv2
from cv_bridge import CvBridge, CvBridgeError

class DepthCoordinatePublisher:
    def __init__(self):
        rospy.init_node('depth_coordinate_publisher', anonymous=True)

        # Define the (x, y) coordinate (adjustable)
        self.xy_coordinate = (320, 240)  # Within 640x400 for LIMO

        # Set up publishers and subscribers
        self.depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        self.depth_pub = rospy.Publisher('/depth_at_coordinate', Float32, queue_size=10)

        # Bridge to convert ROS Image messages to OpenCV images
        self.bridge = CvBridge()
    def depth_callback(self, data):
        try:
            # Convert the depth image to a CV2-compatible image
            depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')

            # Get the dimensions of the image
            # height, width = depth_image.shape
            # rospy.loginfo(f"Depth image dimensions: {width}x{height}")

            # Check if the coordinate is within bounds
            x, y = self.xy_coordinate
            if 0 <= x < width and 0 <= y < height:
                depth_value = depth_image[y, x]
                self.depth_pub.publish(depth_value)
                rospy.loginfo(f"Depth at ({x}, {y}): {depth_value}")
            else:
                rospy.logerr(f"Coordinate ({x}, {y}) is out of bounds!")

        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")

if __name__ == '__main__':
    try:
        DepthCoordinatePublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
