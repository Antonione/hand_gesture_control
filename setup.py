from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'hand_gesture_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Incluindo o diretório de lançamento
        ('share/' + package_name + '/launch', glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anonione',
    maintainer_email='antonione@gmail.com',
    description='Pacote para controle de gestos de mão com Mediapipe e ROS2',
    license='MIT',  # Ou outra licença, conforme necessário
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hand_gesture_detection = hand_gesture_control.hand_gesture_detection:main',
        ],
    },
)
