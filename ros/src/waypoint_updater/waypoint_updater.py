#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
from std_msgs.msg import Int32
import math
import numpy as np

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

'''

LOOKAHEAD_WPS = 30 # Number of waypoints we will publish. You can change this number
MAX_DECEL = .5      # Maximum deceleration for vehicle.

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.pose = None
        self.base_waypoints = None
        self.stopline_wp_idx = -1
        self.waypoints_2d = None
        self.waypoint_tree = None

        self.velocity = self.vel2mps(rospy.get_param('/waypoint_loader/velocity'))
        self.wheel_base = rospy.get_param('~wheel_base', 2.8498 )

        rospy.loginfo("Wheel base: {}".format(self.wheel_base))

        #rospy.spin()
        self.loop()

    def loop(self):
        rate = rospy.Rate( 50 )
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints:
                self.publish_waypoints()
            rate.sleep()

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y

        closest_idx = self.waypoint_tree.query([x,y], 1)[1]
        #Check is closest ahead or behind
        closest_coord = self.waypoints_2d[ closest_idx ]
        prev_coord = self.waypoints_2d[ closest_idx - 1 ]

        cl_vect = np.array( closest_coord )
        prev_vect = np.array( prev_coord )
        pos_vect = np.array( [x,y] )

        val = np.dot( cl_vect - prev_vect, pos_vect - cl_vect )
        if  val > 0:
            closest_idx = (closest_idx + 1 ) % len( self.waypoints_2d )
        return closest_idx

    def publish_waypoints(self):
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish( final_lane ) 

    def generate_lane(self):
        lane = Lane()

        closest_idx = self.get_closest_waypoint_idx()
        farest_idx = closest_idx + LOOKAHEAD_WPS

        base_wp = self.base_waypoints.waypoints[closest_idx:farest_idx]

        #rospy.loginfo("generate line. Stopline idx: {}; closest idx: {}; farest_idx: {}".format( self.stopline_wp_idx, closest_idx, farest_idx ))

        if self.stopline_wp_idx == -1 or ( self.stopline_wp_idx > farest_idx ):
            lane.waypoints = base_wp
        else:
            # If we detect RED Traffic Light we try to decelerate
            lane.waypoints = self.decelerate_waypoints( base_wp, closest_idx )

        return lane

    def decelerate_waypoints(self, waypoints, closest_idx ):
        new_wp = []
        for i, wp in enumerate(waypoints):

            p = Waypoint()
            p.pose = wp.pose

            stop_idx = max( self.stopline_wp_idx - closest_idx - 2, 0 )
            dist = max(self.distance( waypoints, i, stop_idx ) - self.wheel_base, 0)
            vel = math.sqrt( 2 * MAX_DECEL * dist )
            rospy.loginfo("Decelerate waypoints. Stop idx: {0}; distance: {1}; velocity: {2}".format( stop_idx, dist, vel ))
            if vel < 1.0:
                vel = 0

            p.twist.twist.linear.x = min( vel, wp.twist.twist.linear.x)
            new_wp.append(p)

        return new_wp


    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints ]
            self.waypoint_tree = KDTree( self.waypoints_2d )

    def traffic_cb(self, msg):
        self.stopline_wp_idx = msg.data
        if self.stopline_wp_idx > 0:
           rospy.loginfo("Detect stopline at {} waypoint".format(self.stopline_wp_idx))

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def vel2mps(self, velocity):
        return velocity / 3.6


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
