from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    player_0_name = LaunchConfiguration("player_0_name")
    player_1_name = LaunchConfiguration("player_1_name")

    return LaunchDescription(
        [
            DeclareLaunchArgument("player_0_name", default_value="Alice"),
            DeclareLaunchArgument("player_1_name", default_value="Bob"),
            Node(
                package="ttt_referee",
                executable="ttt_referee_node",
                name="ttt_referee",
                output="screen",
            ),
            Node(
                package="ttt_engine",
                executable="ttt_engine_node",
                name="ttt_engine",
                output="screen",
            ),
            Node(
                package="ttt_visualizer",
                executable="ttt_visualizer_node",
                name="ttt_visualizer",
                output="screen",
            ),
            Node(
                package="ttt_player",
                executable="student_player_node",
                name="player_0",
                output="screen",
                parameters=[
                    {
                        "player_name": player_0_name,
                        "plan_turn_service": "/player_0/plan_turn",
                    }
                ],
            ),
            Node(
                package="ttt_player",
                executable="student_player_node",
                name="player_1",
                output="screen",
                parameters=[
                    {
                        "player_name": player_1_name,
                        "plan_turn_service": "/player_1/plan_turn",
                    }
                ],
            ),
        ]
    )
