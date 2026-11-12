"""robot_localization on the sim's GNSS: scripts/gnss_map_odom.py (fix -> map odometry through the map datum) + a
map-frame EKF (config/gnss_ekf.yaml).

  QUADSIM_G23=1 docker/run.sh "ros2 launch launch/gnss_ekf.launch.py"     # with the sim and the nav stack running
  ... "ros2 launch launch/gnss_ekf.launch.py scene:=scenes/osm_tucheng"   # the sim's scene, when not config's
  QUADSIM_G23=1 docker/run.sh "python3 scripts/gnss_ekf_check.py"         # then: drive and compare against ground truth

/odometry/gps is the fix in map coordinates, /odometry/filtered (frame map) the fused pose. The package's own
navsat_transform_node is not used: its datum is anchored to the robot's start pose (see gnss_map_odom.py).
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS = os.path.join(HERE, "..", "config", "gnss_ekf.yaml")


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("scene", default_value="", description="the sim's scene (datum from its geo); '' = config"),
        ExecuteProcess(cmd=["python3", os.path.join(HERE, "..", "scripts", "gnss_map_odom.py"),
                            "--scene", LaunchConfiguration("scene")], output="screen"),
        Node(package="robot_localization", executable="ekf_node", name="ekf_map", output="screen",
             parameters=[PARAMS], remappings=[("odometry/filtered", "/odometry/filtered")]),
    ])
