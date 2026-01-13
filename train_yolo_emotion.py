from ultralytics import YOLO
import os

# ----------------------------------------------------------------------
# Definición de Parámetros
# ----------------------------------------------------------------------

EXPERIMENT_NAME = 'yolov8_emotion'
DATASET_YAML_PATH = 'emotions.yaml' 

# Epocas y Batch_size
NUM_EPOCHS = 100 
BATCH_SIZE = 16 

# Arquitectura para entrenar
ARCHITECTURE_CONFIG = 'yolov8s.pt' 

# ----------------------------------------------------------------------
# Inicialización y Entrenamiento del Modelo
# ----------------------------------------------------------------------

print(f"Cargando la estructura del modelo: {ARCHITECTURE_CONFIG}")
# Cargar la arquitectura.
model = YOLO(ARCHITECTURE_CONFIG)

print("Iniciando entrenamiento para la detección de emociones...")

results = model.train(
    data=DATASET_YAML_PATH,  
    epochs=NUM_EPOCHS,       
    batch=BATCH_SIZE,        
    imgsz=640,               
    name=EXPERIMENT_NAME, 
    lr0=0.001, # Tasa de aprendizaje inicial
    lrf=0.0001, # Tasa de aprendizaje final 
    project='/mnt/homeGPU/pablomarpa/experimentos_yolo',
    augment=True,
    optimizer='SGD'
)

print("Entrenamiento finalizado")
print(f"Resultados guardados en: {os.path.join(os.getcwd(), 'runs', 'detect', EXPERIMENT_NAME)}")
