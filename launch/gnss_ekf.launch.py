"""robot_localization on the sim's GNSS: navsat_transform_node + a map-frame EKF (config/gnss_ekf.yaml).

  docker/run.sh "ros2 launch launch/gnss_ekf.launch.py"      # with the sim and the nav stack running
  docker/run.sh "python3 scripts/gnss_ekf_check.py"          # then: drive and compare against ground truth

Node names match the parameter file's keys. /odometry/filtered (frame map) is the fused pose; /odometry/gps is
the fix in map coordinates; /gnss/filtered is the fused pose back as lat / lon.
"""
import os

from launch import LaunchDescription
from launch_ros.actions import Node

PARAMS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "gnss_ekf.yaml")


def generate_launch_description():
    return LaunchDescription([
        Node(package="robot_localization", executable="navsat_transform_node", name="navsat_transform",
             output="screen", parameters=[PARAMS],
             remappings=[("gps/fix", "/robot01/gnss/fix"), ("odometry/filtered", "/odometry/filtered"),
                         ("odometry/gps", "/odometry/gps"), ("gps/filtered", "/gnss/filtered")]),
        Node(package="robot_localization", executable="ekf_node", name="ekf_map", output="screen",
             parameters=[PARAMS], remappings=[("odometry/filtered", "/odometry/filtered")]),
    ])
