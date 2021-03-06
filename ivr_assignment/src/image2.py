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
    # initialize publishers for joint angles and joint positions
    self.yellow_pos_pub = rospy.Publisher("joints/yellow_pos", Float64MultiArray, queue_size=10)
    self.blue_pos_pub = rospy.Publisher("joints/blue_pos", Float64MultiArray, queue_size=10)
    self.green_pos_pub = rospy.Publisher("joints/green_pos", Float64MultiArray, queue_size=10)
    self.red_pos_pub = rospy.Publisher("joints/red_pos", Float64MultiArray, queue_size=10)
    self.control_angles_pub = rospy.Publisher("control_angles", Float64MultiArray, queue_size=10)
    # initialize a publisher for the position of the target
    self.target_pos_pub = rospy.Publisher("target_im2", Float64MultiArray, queue_size=10)
    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()
    # get times
    self.time_trajectory = rospy.get_time()

  # assigns angles to each joint according to formulae in the coursework doc
  def joint_trajectories(self):
    # gets the current time
    cur_time = np.array([rospy.get_time() - self.time_trajectory])
    joint1 = np.array([0.0])
    joint2 = (np.pi/2) * np.sin( (np.pi/15) * cur_time)
    joint3 = (np.pi/2) * np.sin( (np.pi/18) * cur_time)
    # changed to avoid hitting the ground
    joint4 = (np.pi/3) * np.sin( (np.pi/20) * cur_time)

    self.control_angles.data = np.array([joint1, joint2, joint3, joint4])

  # publishes the calculated angles into the topics used to control robot movement
  def publish_joint_trajectories(self):
    self.robot_joint1_pub.publish(self.control_angles.data[0])
    self.robot_joint2_pub.publish(self.control_angles.data[1])
    self.robot_joint3_pub.publish(self.control_angles.data[2])
    self.robot_joint4_pub.publish(self.control_angles.data[3])
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
    if moments['m00'] == 0:
      temp = self.detect_blue()
      return [0.0, temp[1]]
    else:
      x = int(moments['m10']/moments['m00'])
      y = int(moments['m01']/moments['m00'])
    
    return np.array([x,y])
    
  def detect_red(self):
    cv_image_hsv = cv2.cvtColor(self.cv_image2, cv2.COLOR_BGR2HSV)
    joint = cv2.inRange(cv_image_hsv, (0,50, 50), (5,255,255))
    kernel = np.ones((5,5), np.uint8)
    joint = cv2.dilate(joint, kernel, iterations = 3)
    moments = cv2.moments(joint)
    if moments['m00'] == 0:
      return self.detect_green()
    else:
      x = int(moments['m10']/moments['m00'])
      y = int(moments['m01']/moments['m00'])
    
    return np.array([x,y])

  # takes the length of the first link in pixels as it doesn't move and uses
  # the given length to get a conversion factor
  def pixel_to_meter(self, yellow_centre, blue_centre):
    link_length_p = np.linalg.norm(blue_centre - yellow_centre)
    con_factor = 2.5/link_length_p
    return con_factor

  # takes the coordinates of the centres in pixels and converts them to meters for
  # better angle estimation
  def convert_centres(self, centres_p):
    centres_m = []
    # set joint 1 as the centre of the image
    centres_m.append([0,0])
    con_factor = self.pixel_to_meter(centres_p[0], centres_p[1])

    # for each joint after the first, get pixel values relative to joint 1 and convert to metres
    for centre in centres_p[1:]:
        x = (centre[0] - centres_p[0][0])*con_factor
        # y pixel value will always be greater than other joints so subract from that to get y axis pointing up
        z = (centres_p[0][1] - centre[1])*con_factor
        centres_m.append(np.array([x,z]))
    
    return centres_m

  # 2.2 -----------------

  # find the orange sphere in the image
  def find_target(self):
    # turn image to hsv and isolate the orange colour
    cv_image_hsv = cv2.cvtColor(self.cv_image2, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(cv_image_hsv, (10, 0, 0), (20, 255, 255))
    # get a grayscale version of the image to feed to HoughCircles and apply the mask to it
    grey = cv2.cvtColor(self.cv_image2, cv2.COLOR_BGR2GRAY)
    isolated_grey = cv2.bitwise_and(mask, grey)
    # find circles in the image based on the given parameters
    circles = cv2.HoughCircles(isolated_grey, cv2.HOUGH_GRADIENT, 1.2, 100, param1=50, param2=10, minRadius=10, maxRadius=15)

    # get the centre of each circle found, should only be one in theory
    centres = []
    if circles is not None:
      circles = np.round(circles[0, :]).astype("int")
      for (x, y, r) in circles:
        centres.append(np.array([x,y]))
    return centres
  
  # get the coordinates of the circle in meters as before
  def target_to_meters(self, target):
    image_centre = self.detect_yellow()
    blue_centre = self.detect_blue()
    con_factor = self.pixel_to_meter(image_centre, blue_centre)
    x = target[0] - image_centre[0]
    z = image_centre[1] - target[1]
    self.target_metres = np.array([x,z])*con_factor
  
  # publish the position to a topic to be read by image1
  def publish_target(self):
    self.target_pos = Float64MultiArray()
    self.target_pos.data = [0, 0]
    self.target_pos.data = self.target_metres
    self.target_pos_pub.publish(self.target_pos)
    

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

    # find and publish the target coords
    circle_centres = self.find_target()
    self.target_to_meters(circle_centres[0])
    self.publish_target()

    # setup arrays for publishing
    self.yellow_pos = Float64MultiArray()
    self.blue_pos = Float64MultiArray()
    self.green_pos = Float64MultiArray()
    self.red_pos = Float64MultiArray()

    self.yellow_pos.data = np.array([0.0, 0.0])
    self.blue_pos.data = np.array([0.0, 0.0])
    self.green_pos.data = np.array([0.0, 0.0])
    self.red_pos.data = np.array([0.0, 0.0])

    # get joint centres
    yellow = self.detect_yellow()
    blue = self.detect_blue()
    green = self.detect_green()
    red = self.detect_red()

    # get the intial joint positions
    centres_p = [yellow, blue, green, red]
    # get the positions for publishing
    [self.yellow_pos.data, self.blue_pos.data, self.green_pos.data, self.red_pos.data] = self.convert_centres(centres_p)


    im2=cv2.imshow('window2', self.cv_image2)
    cv2.waitKey(1)

    # Publish the results
    try: 
      self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.cv_image2, "bgr8"))
      # publish the joints to the joints_pos2 topic
      self.yellow_pos_pub.publish(self.yellow_pos)
      self.blue_pos_pub.publish(self.blue_pos)
      self.green_pos_pub.publish(self.green_pos)
      self.red_pos_pub.publish(self.red_pos)
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


