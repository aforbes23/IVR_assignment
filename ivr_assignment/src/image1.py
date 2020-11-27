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
    self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
    self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
    self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
    self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)
    # initialize subscirbers for joint positions from image2
    self.yellow_pos_sub = rospy.Subscriber("joints/yellow_pos", Float64MultiArray, self.yellow_listener)
    self.blue_pos_sub = rospy.Subscriber("joints/blue_pos", Float64MultiArray, self.blue_listener)
    self.green_pos_sub = rospy.Subscriber("joints/green_pos", Float64MultiArray, self.green_listener)
    self.red_pos_sub = rospy.Subscriber("joints/red_pos", Float64MultiArray, self.red_listener)
    # initialise a subscriber for the target coords from image2 and publisher for final coords
    self.im2_target_sub = rospy.Subscriber("target_im2", Float64MultiArray, self.target_listener)
    self.target_estimate_pub = rospy.Publisher("target_estimate", Float64MultiArray, queue_size=10)
    # initialize a publisher for final joint angle estimate output
    self.estimated_angles_pub = rospy.Publisher("estimated_joints", Float64MultiArray, queue_size=10)
    # intialize a publisher for the forward kinematics end effector estimate
    self.fk_pub = rospy.Publisher("fk_estimate", Float64MultiArray, queue_size=10)
    self.end_effector_pub = rospy.Publisher("end_effector_pos", Float64MultiArray, queue_size=10)
    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()
    # get times
    self.time_trajectory = rospy.get_time()
    self.first_time = rospy.get_time()

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
    if moments['m00'] == 0:
      return self.detect_blue()
    else:
      x = int(moments['m10']/moments['m00'])
      y = int(moments['m01']/moments['m00'])
    
    return np.array([x,y])
    

  def detect_red(self):
    cv_image_hsv = cv2.cvtColor(self.cv_image1, cv2.COLOR_BGR2HSV)
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
    centres_m.append(np.array([0,0]))
    con_factor = self.pixel_to_meter(centres_p[0], centres_p[1])

    # for each joint after the first, get pixel values relative to joint 1 and convert to metres
    # y values should be negative as they shouldn't be below joint 1
    for centre in centres_p[1:]:
        y = (centre[0] - centres_p[0][0])*con_factor
        # y pixel value will always be greater than other joints so subract from that to get y axis pointing up
        z = (centres_p[0][1] - centre[1])*con_factor
        centres_m.append(np.array([y,z]))
    
    return np.array(centres_m)


  # uses the data from image2 to get the positions of each joint in the 3d space
  def get_3d_positions(self):
    joints = []
    # for each joint, get the x position from im2, the y position from im1 and the z position from either
    # as it will be the same in both
    for i in range(self.im1_positions.shape[0]):
      x = self.im2_positions[i][0]
      y = self.im1_positions[i][0]
      if np.isnan(self.im2_positions[i][1]):
        z = self.im1_positions[i][1]
      else:
        z = self.im2_positions[i][1]
      joints.append(np.array([x,y,z]))

    return np.array(joints)
    
  def get_angles(self):
    angles = []
    joints = self.joints_pos
    # joint1 fixed in this case
    angles.append(0.0)
    # joint2 is taken from the opp and adjacent on the y,z plane as it rotates around x
    opp = joints[2][1] - joints[1][1]
    adj = joints[2][2] - joints[1][2]
    # take the negative of the angle as the direction is flipped comapred to joint3
    angles.append(-np.arctan2(opp, adj))
    # joint3 is taken from the opp and adjacent on the x,z plane as it rotates around y
    opp = joints[2][0] - joints[1][0]
    adj = joints[2][2] - joints[1][2]
    angles.append(np.arctan2(opp, adj))

    # get rotation matrix for frame after first rotations around x and y
    gamma = angles[1]
    beta = angles[2]
    rotation = np.array([
      [np.cos(beta), 0, np.sin(beta)],
      [-np.sin(gamma)*np.sin(beta), np.cos(gamma), -np.sin(gamma)*np.sin(beta)],
      [np.cos(gamma)*np.sin(beta), np.sin(gamma), np.cos(beta)*np.cos(gamma)]
    ])
    # get vectors for y and z axis, don't need x to get the angle
    y_axis = np.matmul(np.transpose(np.array([0,1,0])), rotation)
    z_axis = np.matmul(np.transpose(np.array([0,0,1])), rotation)
    # get transformed vector from green to red
    g_r_vector = joints[3] - joints[2]
    transformed_vector = np.matmul(np.transpose(g_r_vector), rotation)
    # joint4 is calculated on the opp and adjacent on it's plane of movement, neither xz nor yz
    # get distance between red and green on the x,y plane
    opp = transformed_vector[1]
    adj = transformed_vector[2]

    # joint movement is negative of rotation angle to bring frame to the vector
    angles.append(-np.arctan2(opp, adj))

    return np.array(angles)

  # find the orange sphere in the image
  def find_target(self):
    # turn image to hsv and isolate the orange colour
    cv_image_hsv = cv2.cvtColor(self.cv_image1, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(cv_image_hsv, (10, 0, 0), (20, 255, 255))
    # get a grayscale version of the image to feed to HoughCircles and apply the mask to it
    grey = cv2.cvtColor(self.cv_image1, cv2.COLOR_BGR2GRAY)
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
    y = target[0] - image_centre[0]
    z = image_centre[1] - target[1]
    self.im1_target = np.array([y,z])*con_factor

  # as with the other joints, get the 3d position of the sphere using the data from image2
  def get_3d_target_pos(self):
    x = self.im2_target[0]
    y = self.im1_target[0]
    if np.isnan(self.im2_target[1]):
      z = self.im1_target[1]
    else:
      z = self.im2_target[1]
    return np.array([x,y,z])


  def forward_kinematics(self):
    [p1, a2, a3, a4] = self.estimated_angles.data

    self.end_effector_fk = np.array([
      3*np.sin(a3)*np.cos(a4)*np.cos(p1) + 3.5*np.sin(a3)*np.cos(p1) - 3*np.sin(a4)*np.cos(a2)*np.sin(p1) - 3*np.cos(a3)*np.cos(a4)*np.sin(p1)*np.sin(a2) - 3.5*np.cos(a3)*np.sin(p1)*np.sin(a2),
      -3*np.sin(a3)*np.cos(a4)*np.sin(p1) - 3.5*np.sin(a3)*np.sin(p1) - 3*np.sin(a4)*np.cos(p1)*np.cos(a2) - 3*np.cos(a3)*np.cos(a4)*np.cos(p1)*np.sin(a2) - 3.5*np.cos(a3)*np.cos(p1)*np.sin(a2),
      -3*np.sin(a4)*np.sin(a2) + 3*np.cos(a3)*np.cos(a4)*np.cos(a2) + 3.5*np.cos(a3)*np.cos(a2) + 2.5
    ])

  def open_loop(self):
    target = self.target_estimate
    [a1, a2, a3, a4] = self.estimated_angles.data

    jacobian = np.array([
      [
        -3*np.sin(a3)*np.cos(a4)*np.sin(a1) - 3.5*np.sin(a3)*np.sin(a1) -3*np.sin(a4)*np.cos(a2)*np.cos(a1) - 3*np.cos(a3)*np.cos(a4)*np.cos(a1)*np.sin(a2) - 3.5*np.cos(a3)*np.cos(a1)*np.sin(a2),
        3*np.sin(a4)*np.sin(a2)*np.sin(a1) - 3*np.cos(a3)*np.cos(a4)*np.sin(a1)*np.cos(a2) - 3.5*np.cos(a3)*np.sin(a1)*np.cos(a2),
        3*np.cos(a3)*np.cos(a4)*np.cos(a1) + 3.5*np.cos(a3)*np.cos(a1) + 3*np.sin(a3)*np.cos(a4)*np.sin(a1)*np.sin(a2) + 3.5*np.sin(a3)*np.sin(a1)*np.sin(a2),
        -3*np.sin(a3)*np.sin(a4)*np.cos(a1) - 3*np.cos(a4)*np.cos(a2)*np.sin(a1) + 3*np.cos(a3)*np.sin(a4)*np.sin(a1)*np.sin(a2)
      ],
      [
        -3*np.sin(a3)*np.cos(a4)*np.cos(a1) - 3.5*np.sin(a3)*np.cos(a1) + 3*np.sin(a4)*np.sin(a1)*np.cos(a2) + 3*np.cos(a3)*np.cos(a4)*np.sin(a1)*np.sin(a2) + 3.5*np.cos(a3)*np.sin(a1)*np.sin(a2),
        3*np.sin(a4)*np.cos(a1)*np.sin(a2) - 3*np.cos(a3)*np.cos(a4)*np.cos(a1)*np.cos(a2) - 3.5*np.cos(a3)*np.cos(a1)*np.cos(a2),
        -3*np.cos(a3)*np.cos(a4)*np.sin(a1) - 3.5*np.cos(a3)*np.sin(a1) + 3*np.sin(a3)*np.cos(a4)*np.cos(a1)*np.sin(a2) + 3.5*np.sin(a3)*np.cos(a1)*np.sin(a2),
        3*np.sin(a3)*np.sin(a4)*np.sin(a1) - 3*np.cos(a4)*np.cos(a1)*np.cos(a2) + 3*np.cos(a3)*np.sin(a4)*np.cos(a1)*np.sin(a2)
      ],
      [
        0,
        -3*np.sin(a4)*np.cos(a2) - 3*np.cos(a3)*np.cos(a4)*np.sin(a2) - 3.5*np.cos(a3)*np.sin(a2),
        -3*np.sin(a3)*np.cos(a4)*np.cos(a2) - 3.5*np.sin(a3)*np.cos(a2),
        -3*np.cos(a4)*np.sin(a2) - 3*np.cos(a3)*np.sin(a4)*np.cos(a2)
      ]
    ])

    pinv_jac = np.linalg.pinv(jacobian)
    current_time = rospy.get_time()
    dt = current_time - self.first_time
    self.first_time = current_time

    x_v = (target - self.joints_pos[3])/dt
    q_v = np.dot(pinv_jac,x_v)

    q = np.array([a1, a2, a3, a4]) + dt*q_v
    self.publish_joint_instructions(q)
    

  def publish_joint_instructions(self, angles):
    self.robot_joint1_pub.publish(angles[0])
    self.robot_joint2_pub.publish(angles[1])
    self.robot_joint3_pub.publish(angles[2])
    self.robot_joint4_pub.publish(angles[3])

  def publish_fk(self):
    self.fk_estimate = Float64MultiArray()
    self.fk_estimate.data = self.end_effector_fk
    self.fk_pub.publish(self.fk_estimate)

  def publish_end_effector(self):
    end_effector = Float64MultiArray()
    end_effector.data = self.joints_pos[3]
    self.end_effector_pub.publish(end_effector)
  
  # publish the position to a suitable topic
  def publish_target(self):
    target_pos = Float64MultiArray()
    target_pos.data = [0,0,0]
    target_pos.data = self.target_estimate
    self.target_estimate_pub.publish(target_pos)
  
  # Recieve positions of joints from image2 in 4 methods below
  def yellow_listener(self, arr):
    self.im2_yellow = np.array(arr.data)

  def blue_listener(self, arr):
    self.im2_blue = np.array(arr.data)
  
  def green_listener(self, arr):
    self.im2_green = np.array(arr.data)
  
  def red_listener(self, arr):
    self.im2_red = np.array(arr.data)
  
  # get the xz coordinates of the target from image2
  def target_listener(self, coords):
    self.im2_target = np.array(coords.data)


  # Recieve data from camera 1, process it, and publish
  def callback1(self,data):
    # Recieve the image
    try:
      self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    # find im1 target coordines
    circle_centres = self.find_target()
    self.target_to_meters(circle_centres[0])

    # get and publish final target estimate
    self.target_estimate = self.get_3d_target_pos()
    self.publish_target()

    # get joint coordinates
    yellow = self.detect_yellow()
    blue = self.detect_blue()
    green = self.detect_green()
    red = self.detect_red()
    

    # get intial joint angles
    centres_p = [yellow, blue, green, red]
    self.im1_positions = self.convert_centres(centres_p)
    self.im2_positions = np.array([self.im2_yellow, self.im2_blue, self.im2_green, self.im2_red])

    # get joint positions
    self.joints_pos = self.get_3d_positions()



    # find final joint angles and put them in a variable for publishing
    self.estimated_angles = Float64MultiArray()
    self.estimated_angles.data = np.array([0.0,0.0,0.0,0.0])
    self.estimated_angles.data = self.get_angles()

    # get the end_effector from forward kinematics
    self.forward_kinematics()
    self.publish_fk()
    self.publish_end_effector()

    # get the robot to follow the target
    self.open_loop()
    
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

