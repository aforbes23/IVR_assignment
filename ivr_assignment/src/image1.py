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
    # initialize a publisher to send images from camera1 to a topic named image_topic1
    self.image_pub1 = rospy.Publisher("image_topic1",Image, queue_size = 1)
    # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
    self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw",Image,self.callback1)
    # initialize a publisher for the robot joint controllers
    self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
    self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
    self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)
    # initialize a subscirber for joint angles from image2
    self.joint_angles_sub = rospy.Subscriber("/joints2/", Float64MultiArray, self.angle_listener)
    # initialize a publisher for final joint angle estimate output
    self.estimated_angles_pub = rospy.Publisher("estimated_joints", Float64MultiArray, queue_size=10)
    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()
    # get times
    self.time_trajectory = rospy.get_time()

  # detects the yellow joint at the base of the robot and assigns coordinates to it
  def detect_yellow(self):
    cv_image_hsv = cv2.cvtColor(self.cv_image1, cv2.COLOR_BGR2HSV)
    joint = cv2.inRange(cv_image_hsv, (24,50,50), (34,255,255))
    kernel = np.ones((5,5), np.uint8)
    joint = cv2.dilate(joint, kernel, iterations = 3)
    moments = cv2.moments(joint)
    x = int(moments['m10']/moments['m00'])
    y = int(moments['m01']/moments['m00'])
    
    return np.array([x,y])

  # same as yellow with different color values for the rest
  def detect_blue(self):
    cv_image_hsv = cv2.cvtColor(self.cv_image1, cv2.COLOR_BGR2HSV)
    joint = cv2.inRange(cv_image_hsv, (97,50,50), (121,255,255))
    kernel = np.ones((5,5), np.uint8)
    joint = cv2.dilate(joint, kernel, iterations = 3)
    moments = cv2.moments(joint)
    x = int(moments['m10']/moments['m00'])
    y = int(moments['m01']/moments['m00'])
    
    return np.array([x,y])

  def detect_green(self):
    cv_image_hsv = cv2.cvtColor(self.cv_image1, cv2.COLOR_BGR2HSV)
    joint = cv2.inRange(cv_image_hsv, (40,50,50), (63,255,255))
    kernel = np.ones((5,5), np.uint8)
    joint = cv2.dilate(joint, kernel, iterations = 3)
    moments = cv2.moments(joint)
    x = int(moments['m10']/moments['m00'])
    y = int(moments['m01']/moments['m00'])
    
    return np.array([x,y])
    

  def detect_red(self):
    cv_image_hsv = cv2.cvtColor(self.cv_image1, cv2.COLOR_BGR2HSV)
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
    # y values should be negative as they shouldn't be below joint 1
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
      angle = np.arctan2(opp, adj)
      angles.append(angle)
  
    return np.array(angles)
  
  # defines the angle of the joint in it's own frame
  def find_final_angles(self):
    # no need to use angles[0] as it is always 0 in this case
    # for 1 and 2, use the angle calculate in each of the cameras on their respective rotation axis
    joint2 = self.im1_angles[1]
    joint3 = self.im2_angles[1]
    # add together the angles observed in each frame as it will be rotated depending on 2 and 3
    # take away 2 and 3 to take it back to it's own frame
    joint4 = self.im1_angles[2] + self.im2_angles[2] - joint2 - joint3
    return np.array([joint2, joint3, joint4])

  
  # Recieve angles from image2.py and assigns them to a variable for use in method above
  def angle_listener(self, arr):
    self.im2_angles = np.array(arr.data)


  # Recieve data from camera 1, process it, and publish
  def callback1(self,data):
    # Recieve the image
    try:
      self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    # get joint coordinates
    yellow = self.detect_yellow()
    blue = self.detect_blue()
    green = self.detect_green()
    red = self.detect_red()

    # get intial joint angles
    centres_p = [yellow, blue, green, red]
    centres_m = self.convert_centres(centres_p)
    self.im1_angles = self.find_angles(centres_m)

    # find final joint angles and put them in a variable for publishing
    self.estimated_angles = Float64MultiArray()
    self.estimated_angles.data = np.array([0.0,0.0,0.0])
    self.estimated_angles.data = self.find_final_angles()


    
    # Uncomment if you want to save the image
    #cv2.imwrite('image_copy.png', cv_image)

    im1=cv2.imshow('window1', self.cv_image1)
    cv2.waitKey(1)
    # Publish the results
    try: 
      self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
      # publish the final angles to the estimated_joints topic
      self.estimated_angles_pub.publish(self.estimated_angles)
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


