import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String  # Corrigido para std_msgs.msg
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
        # Publisher para o gesto detectado
        self.gesture_publisher = self.create_publisher(String, '/hand_gesture', 10)
        self.bridge = CvBridge()
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,       # Modo dinâmico para vídeos (False)
            max_num_hands=1,               # Número máximo de mãos que deseja detectar
            min_detection_confidence=0.8,  # Confiança mínima para considerar a detecção da mão
            min_tracking_confidence=0.8    # Confiança mínima para considerar o rastreamento da mão
        )
        self.mp_draw = mp.solutions.drawing_utils

    def image_callback(self, msg):
        # Converte a mensagem ROS em imagem OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Inicializa a variável gesture para evitar erro
        gesture = "Nenhum gesto detectado"
        
        # Processa a imagem com o MediaPipe Hands
        results = self.mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Desenha as landmarks da mão
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                gesture = self.detect_gesture(hand_landmarks)

                # Publica o gesto detectado
                self.gesture_publisher.publish(String(data=gesture))
                self.get_logger().info(f"Gesto detectado: {gesture}")
        
        # Checa o parâmetro 'show_window' para determinar se deve exibir a imagem
        show_window = self.get_parameter('show_window').get_parameter_value().bool_value
        if show_window:
            # Adiciona o texto do gesto no frame
            cv2.putText(frame, f"Gesto: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow("Hand Gesture Control", frame)
            cv2.waitKey(1)

    def detect_gesture(self, hand_landmarks):
        """
        Detecta gestos simples com base nas landmarks da mão.
        Exemplos: mão aberta, punho fechado, joinha, V de vitória.
        """
        landmarks = hand_landmarks.landmark

        # Posições chave para análise dos gestos
        thumb_tip_x = landmarks[mp.solutions.hands.HandLandmark.THUMB_TIP].x
        thumb_tip_y = landmarks[mp.solutions.hands.HandLandmark.THUMB_TIP].y
        thumb_ip_y = landmarks[mp.solutions.hands.HandLandmark.THUMB_IP].y
        thumb_mcp_x = landmarks[mp.solutions.hands.HandLandmark.THUMB_MCP].x

        index_finger_tip = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y
        middle_finger_tip = landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].y
        ring_finger_tip = landmarks[mp.solutions.hands.HandLandmark.RING_FINGER_TIP].y
        pinky_finger_tip = landmarks[mp.solutions.hands.HandLandmark.PINKY_TIP].y

        # Posições das juntas MCP (base dos dedos)
        index_finger_mcp = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP].y
        middle_finger_mcp = landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP].y
        ring_finger_mcp = landmarks[mp.solutions.hands.HandLandmark.RING_FINGER_MCP].y
        pinky_finger_mcp = landmarks[mp.solutions.hands.HandLandmark.PINKY_MCP].y

        # Verificação de "punho fechado" (todos os dedos dobrados, polegar incluído)
        if (index_finger_tip > index_finger_mcp and
            middle_finger_tip > middle_finger_mcp and
            ring_finger_tip > ring_finger_mcp and
            pinky_finger_tip > pinky_finger_mcp and
            thumb_tip_x < thumb_mcp_x):
            return "punho fechado"

        # Detecção de "mão aberta" (todos os dedos estendidos)
        if (index_finger_tip < index_finger_mcp and
            middle_finger_tip < middle_finger_mcp and
            ring_finger_tip < ring_finger_mcp and
            pinky_finger_tip < pinky_finger_mcp and
            thumb_tip_y < thumb_ip_y):
            return "mão aberta"

        else:
            return "gesto desconhecido"


def main(args=None):
    rclpy.init(args=args)
    hand_gesture_control = HandGestureControl()
    rclpy.spin(hand_gesture_control)
    hand_gesture_control.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
