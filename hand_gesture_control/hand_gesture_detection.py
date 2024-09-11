import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import mediapipe as mp

class HandGestureControl(Node):
    def __init__(self):
        super().__init__('hand_gesture_control')
        self.declare_parameter('show_window', False)
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10)
        self.bridge = CvBridge()
        self.mp_hands = mp.solutions.hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils

    def image_callback(self, msg):
        # Converte a mensagem ROS em imagem OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Processa a imagem com o MediaPipe Hands
        results = self.mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Desenha as landmarks da mão
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        
        # Checa o parâmetro 'show_window' para determinar se deve exibir a imagem
        show_window = self.get_parameter('show_window').get_parameter_value().bool_value
        if show_window:
            cv2.imshow("Hand Gesture Control", frame)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    hand_gesture_control = HandGestureControl()
    rclpy.spin(hand_gesture_control)
    hand_gesture_control.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
