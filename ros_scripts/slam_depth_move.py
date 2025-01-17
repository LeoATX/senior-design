#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped

class DepthTo2DNavGoal:
    def __init__(self):
        rospy.init_node("depth_to_2d_nav_goal", anonymous=True)

        # Parameters
        self.goal_topic = rospy.get_param("~goal_topic", "/move_base_simple/goal")  # 2D Nav Goal topic
        self.base_frame = rospy.get_param("~base_frame", "base_link")  # Frame of reference (base_link)
        self.xy_coordinate = rospy.get_param("~xy_coordinate", [320, 240])  # Pixel location in the depth image

        # State to track if the goal is already published
        self.goal_published = False

        # Subscriber for the depth value at the specified coordinate
        self.depth_sub = rospy.Subscriber("/depth_at_coordinate", Float32, self.depth_callback)

        # Publisher for the 2D navigation goal
        self.goal_pub = rospy.Publisher(self.goal_topic, PoseStamped, queue_size=10)

        rospy.loginfo("Depth to 2D Navigation Goal initialized.")

    def depth_callback(self, msg):
        # Check if the goal has already been published
        if self.goal_published:
            rospy.loginfo("2D Navigation Goal already published. Ignoring further depth data.")
            return

        # Process the depth value and create the navigation goal
        depth_value = msg.data
        rospy.loginfo(f"Received depth value: {depth_value} mm.")

        if depth_value <= 0:
            rospy.logwarn("Depth value is invalid or zero. Ignoring.")
            return

        # Create a PoseStamped message for the goal
        goal = PoseStamped()
        goal.header.frame_id = 'base_link'  # Set the frame of reference to the map
        goal.header.stamp = rospy.Time.now()

        # Use depth value to set the goal's x position
        goal.pose.position.x = depth_value/1000  # Forward distance in mm
        goal.pose.position.y = 0.0  # Lateral offset (centered)
        goal.pose.position.z = 0.0  # Ground level

        # Set orientation to face forward
        goal.pose.orientation.x = 0.0
        goal.pose.orientation.y = 0.0
        goal.pose.orientation.z = 0.0
        goal.pose.orientation.w = 1.0

        # Publish the goal to RViz
        self.goal_pub.publish(goal)
        rospy.loginfo(f"Published 2D Navigation Goal: x={goal.pose.position.x}, y={goal.pose.position.y}, frame_id={self.base_frame}")

        # Mark the goal as published
        self.goal_published = True

        # Shut down the node after publishing the goal
        rospy.signal_shutdown("2D Navigation Goal published, shutting down.")

if __name__ == "__main__":
    try:
        DepthTo2DNavGoal()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
