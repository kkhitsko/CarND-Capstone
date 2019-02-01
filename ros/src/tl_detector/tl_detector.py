#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image

from light_classification.resnet_tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial import KDTree



STATE_COUNT_THRESHOLD = 2

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.limit = 0
        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.waypoint_tree = None
        self.waypoints_2d = None

	# Subscribers
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        sub3 = rospy.Subscriber('/image_color', Image, self.image_cb)
        sub4 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)

        #self.image_nb_pub = rospy.Publisher('/image_traffic_light', Image, queue_size=1)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.listener = tf.TransformListener()

        # Traffic Light State.
        self.state = TrafficLight.UNKNOWN

        self.last_wp = -1
        self.state_cnt = 0

        self.skip_step = self.config['skip_step']
        self.is_debug = self.config['is_debug']
        self.threshold = self.config['threshold']


        self.light_classifier = TLClassifier(is_debug=self.is_debug,threshold=self.threshold)

        rospy.spin()


    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def get_light_state_string(self, state):
         light_str = "UNKNOWN"
         if state == 0:
             light_str = "RED"
         elif state == 2:
             light_str = "GREEN"
         elif state == 1:
             light_str = "YELLOW"

         return light_str

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        # For perfomance purpose we detect light from every self.skip_step image
        self.limit = (self.limit + 1) % self.skip_step
        if self.limit !=0:
            return


        self.has_image = True
        self.camera_image = msg
        light_wp, cur_state = self.process_traffic_lights()

        if self.is_debug:
            rospy.loginfo("Traffic light waypoint: {0}; state: {1}".format( light_wp, self.get_light_state_string( cur_state) ) )

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != cur_state:
            self.state_cnt = 0
            self.state = cur_state
        elif self.state_cnt >= STATE_COUNT_THRESHOLD:
            if cur_state != TrafficLight.RED:
                light_wp = -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
            if self.is_debug:
                rospy.loginfo("Publish red light at waypoint {}; state {} {}".format(light_wp, cur_state, self.get_light_state_string( cur_state) ))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
            if self.is_debug:
                rospy.loginfo("Publish red light at waypoint {}".format(self.last_wp))
        self.state_cnt += 1

        # Print light state for debugging.
        if self.is_debug:
            rospy.loginfo("TRAFFIC_LIGHT: {}; state count: {}".format(self.get_light_state_string(self.state), self.state_cnt))

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        if not self.waypoint_tree:
            return
        closest_idx = self.waypoint_tree.query([x,y],1)[1]
	
        # Check if closest is ahead or behind vehicle.
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]

        closest_vest = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x,y])

        val = np.dot(closest_vest - prev_vect, pos_vect - closest_vest)
        
        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)

        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")

        return self.light_classifier.get_classification(cv_image)


    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = None

        # List of stop line positions
        stop_line_positions = self.config['stop_line_positions']
        if self.pose:
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)
            #rospy.loginfo("Current position: x {0}, y {1}. Closest wp: {2}".format(self.pose.pose.position.x, self.pose.pose.position.y, car_wp_idx) )
            # Iterate through all intersections to find closest.
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])

                d = temp_wp_idx - car_wp_idx

                if 0 <= d < diff:
                    if self.is_debug:
                        rospy.loginfo("Number of waypoints: {0}; distance to stopline: {1}".format( diff, d ) )
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx

        # If there is an intersection nearby.
        if closest_light or self.is_site:
            state = self.get_light_state(closest_light)
            return line_wp_idx, state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
