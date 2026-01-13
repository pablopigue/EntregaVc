from ultralytics import YOLO
import cv2
import torch
import os

# Configuración de Dispositivo
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Usando: {device.upper()}")

# Cargar el modelo de emociones
emotion_model_path = "./yolov8_emotion/weights/best.pt"
model = YOLO(emotion_model_path).to(device)

# RUTA DE LA IMAGEN Y SALIDA
image_path = "./caras_extraidas/cara_2.jpg"
output_dir = "./emociones_extraidas"
output_path = os.path.join(output_dir, "resultado_emocion.jpg")

# Crear carpeta de salida si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(image_path):
    print(f"Error: No se encontró la imagen en {image_path}")
else:
    # Leer imagen
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    
    # Inferencia
    results = model(img, conf=0.3, device=device, verbose=False)

    # Dibujar los resultados
    for r in results:
        for box in r.boxes:
            # Obtener coordenadas
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Obtener clase y confianza
            cls = int(box.cls[0])
            label_name = r.names[cls].upper()
            conf = float(box.conf[0])

            # Configuración visual
            color = (255, 255, 0) # Cyan
            texto = f"{label_name} {conf:.2f}"
            
            # Calculamos el tamaño del texto para el fondo
            font_scale = 0.2
            thickness = 1
            (t_w, t_h), baseline = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Si no hay espacio arriba, ponemos el texto dentro del cuadro
            text_y = y1 - 10 if y1 > (t_h + 10) else y1 + t_h + 10
            
            # Dibujar fondo negro para el texto
            cv2.rectangle(img, (x1, text_y - t_h - 5), (x1 + t_w, text_y + baseline), (0, 0, 0), -1)
            
            # Dibujar texto
            cv2.putText(img, texto, (x1, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

            # Dibujar rectángulo de la cara
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

    # Mostrar y Guardar
    cv2.imshow("Deteccion de Emociones", img)
    
    cv2.imwrite(output_path, img)
    print(f"Proceso completado. Imagen guardada como: {output_path}")

    print("Presiona cualquier tecla para cerrar la ventana...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()