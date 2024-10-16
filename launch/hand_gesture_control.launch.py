from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Declara o argumento do lançamento
        DeclareLaunchArgument(
            'show_window', 
            description='Determines whether or not to display the OpenCV window'
        ),
        Node(
            package='hand_gesture_control',
            executable='hand_gesture_detection',  # Certifique-se de que este é o nome correto
            name='hand_gesture_control',
            output='screen',
            parameters=[{'show_window': LaunchConfiguration('show_window')}],  # Parâmetro passado via Launch
        ),
        Node(
            package='hand_gesture_control',
            executable='attention_detection',
            name='attention_detection',
            output='screen'
        ),
    ])
