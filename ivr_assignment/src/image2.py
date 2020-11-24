#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError


class image_converter:

  # Defines publisher and subscriber
  def __init__(self):
    # initialize the node named image_processing
    rospy.init_node('image_processing', anonymous=True)
    # initialize a publisher to send images from camera2 to a topic named image_topic2
    self.image_pub2 = rospy.Publisher("image_topic2",Image, queue_size = 1)
    # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
    self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw",Image,self.callback2)
    # initialize a publisher for the robot joint controllers
    self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
    self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
    self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
    self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)
    # initialize publisher and subscriber for joint angles and actual angles
    self.joint_angles_pub = rospy.Publisher("joints2", Float64MultiArray, queue_size=10)
    self.control_angles_pub = rospy.Publisher("control_angles", Float64MultiArray, queue_size=10)
    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()
    # get times
    self.time_trajectory = rospy.get_time()

  # assigns angles to each joint according to formulae in the coursework doc
  def joint_trajectories(self):
    # gets the current time
    cur_time = np.array([rospy.get_time() - self.time_trajectory])
    joint2 = (np.pi/2) * np.sin( (np.pi/15) * cur_time)
    joint3 = (np.pi/2) * np.sin( (np.pi/18) * cur_time)
    # changed to avoid hitting the ground
    joint4 = (np.pi/3) * np.sin( (np.pi/20) * cur_time)

    self.joints.data = np.array([joint2, joint3, joint4])

  # publishes the calculated angles into the topics used to control robot movement
  def publish_joint_trajectories(self):
    self.robot_joint1_pub.publish(0.0)
    self.robot_joint2_pub.publish(self.control_angles.data[0])
    self.robot_joint3_pub.publish(self.control_angles.data[1])
    self.robot_joint4_pub.publish(self.control_angles.data[2])
    self.control_angles_pub.publish(self.control_angles)
    
  # detects the yellow joint at the base of the robot and assigns coordinates to it
  def detect_yellow(self):
    # converts the image to hsv to detect colors more easily
    cv_image_hsv = cv2.cvtColor(self.cv_image2, cv2.COLOR_BGR2HSV)
    joint = cv2.inRange(cv_image_hsv, (24,50,50), (34,255,255))
    kernel = np.ones((5,5), np.uint8)
    joint = cv2.dilate(joint, kernel, iterations = 3)
    moments = cv2.moments(joint)
    x = int(moments['m10']/moments['m00'])
    y = int(moments['m01']/moments['m00'])
    
    return np.array([x,y])

  # all the rest same as yellow
  def detect_blue(self):
    cv_image_hsv = cv2.cvtColor(self.cv_image2, cv2.COLOR_BGR2HSV)
    joint = cv2.inRange(cv_image_hsv, (97,50,50), (121,255,255))
    kernel = np.ones((5,5), np.uint8)
    joint = cv2.dilate(joint, kernel, iterations = 3)
    moments = cv2.moments(joint)
    x = int(moments['m10']/moments['m00'])
    y = int(moments['m01']/moments['m00'])
    
    return np.array([x,y])

  def detect_green(self):
    cv_image_hsv = cv2.cvtColor(self.cv_image2, cv2.COLOR_BGR2HSV)
    joint = cv2.inRange(cv_image_hsv, (40,50,50), (63,255,255))
    kernel = np.ones((5,5), np.uint8)
    joint = cv2.dilate(joint, kernel, iterations = 3)
    moments = cv2.moments(joint)
    x = int(moments['m10']/moments['m00'])
    y = int(moments['m01']/moments['m00'])
    
    return np.array([x,y])
    

  def detect_red(self):
    cv_image_hsv = cv2.cvtColor(self.cv_image2, cv2.COLOR_BGR2HSV)
    joint = cv2.inRange(cv_image_hsv, (0,50, 50), (5,255,255))
    kernel = np.ones((5,5), np.uint8)
    joint = cv2.dilate(joint, kernel, iterations = 3)
    moments = cv2.moments(joint)
    x = int(moments['m10']/moments['m00'])
    y = int(moments['m01']/moments['m00'])
    
    return np.array([x,y])

  # takes the length of the first link in pixels as it doesn't move and uses
  # the given length to get a conversion factor
  def pixel_to_meter(self, yellow_centre, blue_centre):
    link_length_p = np.linalg.norm(blue_centre - yellow_centre)
    con_factor = 3/link_length_p
    return con_factor

  # takes the coordinates of the centres in pixels and converts them to meters for
  # better angle estimation
  def convert_centres(self, centres_p):
    centres_m = []
    # set joint 1 as the centre of the image
    centres_m.append(np.array([0,0]))
    con_factor = self.pixel_to_meter(centres_p[0], centres_p[1])

    # for each joint after the first, get pixel values relative to joint 1 and convert to metres
    for centre in centres_p[1:]:
        x = (centre[0] - centres_p[0][0])*con_factor
        y = (centre[1] - centres_p[0][1])*con_factor
        centres_m.append(np.array([x,y]))
    
    return centres_m

  # calculates the angle for the joint relative to the base frame
  def find_angles(self, centres_m):
    angles = []
    for i in range(1,len(centres_m)):
      opp = centres_m[i-1][0] - centres_m[i][0]
      adj = centres_m[i][1] - centres_m[i-1][1]
      # as y axis is facing away from the camera, take the negative of the angle to use with image1
      angle = -np.arctan2(opp, adj)
      angles.append(angle)
  
    return np.array(angles)


# Recieve data, process it, and publish
  def callback2(self,data):
    # Recieve the image
    try:
      self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    # Uncomment if you want to save the image
    #cv2.imwrite('image_copy.png', cv_image)

    # move the robot
    self.control_angles = Float64MultiArray()
    self.control_angles.data = np.array([0.0,0.0,0.0])
    self.joint_trajectories()
    self.publish_joint_trajectories()

    # setup array for publishing
    self.joints = Float64MultiArray()
    self.joints.data = np.array([0.0, 0.0, 0.0])

    # get joint centres
    yellow = self.detect_yellow()
    blue = self.detect_blue()
    green = self.detect_green()
    red = self.detect_red()

    # get the values to publish
    centres_p = [yellow, blue, green, red]
    centres_m = self.convert_centres(centres_p)
    self.joints.data = self.find_angles(centres_m)

    im2=cv2.imshow('window2', self.cv_image2)
    cv2.waitKey(1)

    # Publish the results
    try: 
      self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.cv_image2, "bgr8"))
      # publish the joints to the joints2 topic
      self.joint_angles_pub.publish(self.joints)
    except CvBridgeError as e:
      print(e)

# call the class
def main(args):
  ic = image_converter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)


