from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='hand_gesture_control',
            executable='hand_gesture_detection',
            name='hand_gesture_control',
            output='screen',
        ),
    ])
