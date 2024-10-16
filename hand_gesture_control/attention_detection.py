import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import mediapipe as mp

class AttentionDetection(Node):
    def __init__(self):
        super().__init__('attention_detection')
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10)
        self.attention_publisher = self.create_publisher(Bool, '/attention_status', 10)
        self.bridge = CvBridge()

        # Inicializando o MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,  # Detectar no máximo 1 rosto
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

    def image_callback(self, msg):
        # Converte a imagem do ROS para OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processa a imagem com MediaPipe Face Mesh
        results = self.mp_face_mesh.process(rgb_frame)
        
        attention = False
        if results.multi_face_landmarks:
            attention = True
            for face_landmarks in results.multi_face_landmarks:
                # Desenha a malha facial com precisão (pontos e conexões)
                self.mp_draw.draw_landmarks(
                    frame, 
                    face_landmarks, 
                    mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    self.drawing_spec,  # Configura espessura e raio dos pontos
                    self.drawing_spec   # Configura espessura das conexões
                )
               
        # Publica se há atenção detectada (True ou False)
        self.attention_publisher.publish(Bool(data=attention))

        self.get_logger().info(f"Atenção detectada: {attention}")
        
        # Exibe o vídeo com a malha facial desenhada
        cv2.imshow("Attention Detection", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    attention_detection = AttentionDetection()
    rclpy.spin(attention_detection)
    attention_detection.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
