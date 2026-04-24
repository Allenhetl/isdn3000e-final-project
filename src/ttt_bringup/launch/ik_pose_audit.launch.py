from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description() -> LaunchDescription:
    bringup_share = get_package_share_directory("ttt_bringup")

    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(bringup_share, "launch", "moveit_ik.launch.py")
                )
            ),
            Node(
                package="ttt_bringup",
                executable="ik_pose_audit",
                name="ik_pose_audit",
                output="screen",
            ),
        ]
    )
