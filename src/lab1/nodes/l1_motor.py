#!/usr/bin/env python3
import rospy
import math
from geometry_msgs.msg import Twist
from std_msgs.msg import String

def publisher_node():
    """TODO: initialize the publisher node here, \
            and publish wheel command to the cmd_vel topic')"""
    cmd_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
    
    freq = 10
    rate = rospy.Rate(freq)
    dt = 1/freq
    
    time = 0
    x_speed = 0.1
    z_speed = 0.15
    t_stage1 = 1/x_speed
    t_stage2 = t_stage1 + (2 * math.pi) / z_speed
    while not rospy.is_shutdown():
        twist = Twist()
        if time <= t_stage1:
            twist.linear.x = x_speed
            twist.angular.z = 0
        elif time > t_stage1 and time <= t_stage2:
            twist.linear.x = 0
            twist.angular.z = z_speed
        else:
            twist.linear.x = 0
            twist.angular.z = 0
        time += dt
        cmd_pub.publish(twist)
        rate.sleep()


def main():
    try:
        rospy.init_node('motor')
        publisher_node()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
