from ultralytics import YOLO
import cv2
import numpy as np
import torch

# -----------------------------------------
# Detectar Dispositivo CUDA o CPU
# -----------------------------------------
# Si hay GPU NVIDIA disponible y PyTorch con CUDA instalado, usa 'cuda'. Si no, 'cpu'.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Dispositivo seleccionado: {device.upper()}")

if device == 'cuda':
    print(f"   Nombre GPU: {torch.cuda.get_device_name(0)}")

# -----------------------------------------
# Cargar modelos
# -----------------------------------------
print("Cargando modelos...")
face_model = YOLO("./yolov8_face/weights/best.pt")
emotion_model = YOLO("./yolov8_emotion/weights/best.pt")

# Mover modelos al dispositivo detectado
face_model.to(device)
emotion_model.to(device)

# -----------------------------------------
# Iniciar captura de video
# -----------------------------------------
cap = cv2.VideoCapture(0) # 0 para webcam, o ruta del archivo

if not cap.isOpened():
    raise ValueError("No se pudo abrir la cámara.")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Iniciando detección. Presiona 'q' para salir.")

# -----------------------------------------
# Bucle de procesamiento
# -----------------------------------------
while True:
    success, frame = cap.read()
    if not success:
        break

    # Detección de caras
    # Pasamos device=device para usar la GPU si se seleccionó
    face_results = face_model(frame, conf=0.5, device=device, verbose=False, stream=True)

    for result in face_results:
        boxes = result.boxes
        for box in boxes:
            # Obtener coordenadas
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Clamping: Asegurar que las coordenadas no salgan de la imagen
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_width, x2), min(frame_height, y2)

            # Recortar la cara
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                continue

            # Detección de emociones sobre el recorte
            emotion_results = emotion_model(face_crop, conf=0.5, device=device, verbose=False)[0]

            if len(emotion_results.boxes) > 0:
                cls = int(emotion_results.boxes.cls[0])
                emotion_name = emotion_results.names[cls]
                conf = float(emotion_results.boxes.conf[0])
                color = (0, 255, 0) # Verde
            else:
                emotion_name = "Neutral/Wait"
                conf = 0.0
                color = (0, 255, 255) # Amarillo

            # Visualización
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{emotion_name} {conf:.2f}"
            # Fondo negro para texto
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + t_size[0], y1), (0,0,0), -1)
            
            cv2.putText(frame, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # -----------------------------------------
    # Mostrar
    # -----------------------------------------
    cv2.imshow("YOLOv8 Real-Time", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
