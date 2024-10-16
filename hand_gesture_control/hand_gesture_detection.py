import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np
from collections import deque


class HandGestureControl(Node):
    def __init__(self):
        super().__init__('hand_gesture_control')
        self.declare_parameter('show_window', False)
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10)
        # Publisher para o gesto detectado
        self.gesture_publisher = self.create_publisher(String, '/hand_gesture', 10)
        self.bridge = CvBridge()
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False, 
            max_num_hands=2, 
            min_detection_confidence=0.8, 
            min_tracking_confidence=0.8
        )
        self.mp_draw = mp.solutions.drawing_utils
        # Fila para o filtro temporal de estabilidade
        self.gesture_history = deque(maxlen=10)

        self.attention_status = False
        
        # Subscreve ao tópico de atenção
        self.attention_sub = self.create_subscription(
            Bool, '/attention_status', self.attention_callback, 10)

    def attention_callback(self, msg):
        self.attention_status = msg.data

    def image_callback(self, msg):
        # Converte a mensagem ROS em imagem OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        gesture = "Nenhum gesto detectado"
        
        # Processa a imagem com o MediaPipe Hands
        results = self.mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Desenha as landmarks da mão e calcula a bounding box e o centro da palma
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Cálculo do centro da palma
                cx, cy = self.calc_palm_moment(frame, hand_landmarks)
                # Cálculo da bounding box
                brect = self.calc_bounding_rect(frame, hand_landmarks)

                self.mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                gesture = self.detect_gesture(hand_landmarks)

                # Desenha a bounding box
                frame = self.draw_bounding_rect(frame, brect)

                # Apenas publica o gesto se a atenção for verdadeira
                if self.attention_status:
                    self.gesture_publisher.publish(String(data=gesture))
                    self.get_logger().info(f"Gesto detectado: {gesture}")
                else:
                    self.get_logger().info("Atenção não detectada. Ignorando gesto.")

        # Filtro temporal para estabilidade
        self.gesture_history.append(gesture)
        stable_gesture = max(set(self.gesture_history), key=self.gesture_history.count)
        
        # Checa o parâmetro 'show_window' para determinar se deve exibir a imagem
        show_window = self.get_parameter('show_window').get_parameter_value().bool_value
        if show_window:
            # Adiciona o texto do gesto no frame
            cv2.putText(frame, f"Gesto: {stable_gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow("Hand Gesture Control", frame)
            cv2.waitKey(1)

    def detect_gesture(self, hand_landmarks):
        """
        Detecta gestos simples com base nas landmarks da mão.
        Exemplos: mão aberta, punho fechado, joinha.
        """
        landmarks = hand_landmarks.landmark
        thumb_tip_x = landmarks[mp.solutions.hands.HandLandmark.THUMB_TIP].x
        thumb_tip_y = landmarks[mp.solutions.hands.HandLandmark.THUMB_TIP].y
        thumb_ip_y = landmarks[mp.solutions.hands.HandLandmark.THUMB_IP].y
        thumb_mcp_x = landmarks[mp.solutions.hands.HandLandmark.THUMB_MCP].x

        index_finger_tip = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y
        middle_finger_tip = landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].y
        ring_finger_tip = landmarks[mp.solutions.hands.HandLandmark.RING_FINGER_TIP].y
        pinky_finger_tip = landmarks[mp.solutions.hands.HandLandmark.PINKY_TIP].y

        index_finger_mcp = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP].y
        middle_finger_mcp = landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP].y
        ring_finger_mcp = landmarks[mp.solutions.hands.HandLandmark.RING_FINGER_MCP].y
        pinky_finger_mcp = landmarks[mp.solutions.hands.HandLandmark.PINKY_MCP].y

        # Punho fechado
        if (index_finger_tip > index_finger_mcp and middle_finger_tip > middle_finger_mcp and
            ring_finger_tip > ring_finger_mcp and pinky_finger_tip > pinky_finger_mcp and
            thumb_tip_x < thumb_mcp_x):
            return "punho fechado"

        # Mão aberta
        if (index_finger_tip < index_finger_mcp and middle_finger_tip < middle_finger_mcp and
            ring_finger_tip < ring_finger_mcp and pinky_finger_tip < pinky_finger_mcp and
            thumb_tip_y < thumb_ip_y):
            return "mão aberta"
        else:
            return "gesto desconhecido"

    def calc_palm_moment(self, image, landmarks):
        """
        Calcula o centro da palma baseado em landmarks da mão.
        """
        image_width, image_height = image.shape[1], image.shape[0]
        palm_array = np.empty((0, 2), int)

        # Usamos as landmarks do pulso e das bases dos dedos
        for index in [0, 1, 5, 9, 13, 17]:
            landmark_x = min(int(landmarks.landmark[index].x * image_width), image_width - 1)
            landmark_y = min(int(landmarks.landmark[index].y * image_height), image_height - 1)
            palm_array = np.append(palm_array, np.array([[landmark_x, landmark_y]]), axis=0)

        M = cv2.moments(palm_array)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = 0, 0

        return cx, cy

    def calc_bounding_rect(self, image, landmarks):
        """
        Calcula a bounding box para as landmarks da mão.
        """
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_array = np.append(landmark_array, np.array([[landmark_x, landmark_y]]), axis=0)

        x, y, w, h = cv2.boundingRect(landmark_array)
        return [x, y, x + w, y + h]

    def draw_bounding_rect(self, image, brect):
        """
        Desenha a bounding box ao redor da mão detectada.
        """
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 2)
        return image


def main(args=None):
    rclpy.init(args=args)
    hand_gesture_control = HandGestureControl()
    rclpy.spin(hand_gesture_control)
    hand_gesture_control.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
