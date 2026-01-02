from ultralytics import YOLO
import cv2
import torch
import os

# ---------------------------------------------------------
# Configuración Inicial y Dispositivo
# ---------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Usando dispositivo: {device.upper()}")

# Crear carpeta para guardar los recortes de caras si no existe
output_folder = "caras_extraidas"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ---------------------------------------------------------
# Cargar el Modelo de Caras
# ---------------------------------------------------------
model_path = "./yolov8_face/weights/best.pt"
face_model = YOLO(model_path).to(device)

# ---------------------------------------------------------
# Ruta de la Imagen y Carga
# ---------------------------------------------------------
image_path = "./ImagenesPruebaManual/imagen0.jpg" 

if not os.path.exists(image_path):
    print(f"Error: No se encontró la imagen en {image_path}")
else:
    img = cv2.imread(image_path)
    # Hacemos una copia para dibujar sin afectar los recortes limpios
    img_dibujo = img.copy()

    # ---------------------------------------------------------
    # Realizar la Detección
    # ---------------------------------------------------------
    results = face_model(img, conf=0.5, device=device, verbose=False)

    # ---------------------------------------------------------
    # Procesar cada Detección
    # ---------------------------------------------------------
    # Usamos enumerate para darle un número único a cada archivo de cara
    boxes = results[0].boxes
    print(f"Se detectaron {len(boxes)} caras.")

    for i, box in enumerate(boxes):
        # Obtener coordenadas
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])

        # Guardamos cada cara por separado
        # Cortamos la zona de la cara de la imagen original
        cara_recorte = img[y1:y2, x1:x2]
        
        if cara_recorte.size > 0:
            nombre_cara = f"{output_folder}/cara_{i}.jpg"
            cv2.imwrite(nombre_cara, cara_recorte)
            print(f"    Cara {i} guardada en: {nombre_cara}")

        # Dibujamos sobre la copia 'img_dibujo'
        cv2.rectangle(img_dibujo, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"Cara {i}: {conf:.2f}"
        cv2.putText(img_dibujo, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ---------------------------------------------------------
    # Mostrar y Guardar Imagen Final
    # ---------------------------------------------------------
    # Guardar la imagen con todos los rectángulos
    resultado_final = "resultado_completo.jpg"
    cv2.imwrite(resultado_final, img_dibujo)
    print(f"\nImagen general guardada como: {resultado_final}")

    # Mostrar en ventana
    cv2.imshow("Deteccion Multiple", img_dibujo)
    print("\nPresiona cualquier tecla para cerrar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()