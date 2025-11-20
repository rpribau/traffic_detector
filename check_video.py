import cv2
import sys

video_path = "/home/rob/Downloads/video_prueba.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: OpenCV (Python) TAMPOCO puede abrir el video: {video_path}")
    sys.exit(1)

ret, frame = cap.read()
if not ret:
    print(f"Error: Se abrió el video pero no se pudo leer ningún frame.")
else:
    print(f"¡Éxito! OpenCV (Python) puede leer el video. Frame size: {frame.shape}")

cap.release()
