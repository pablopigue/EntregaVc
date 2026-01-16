# EntregaVc



## Configuración del entorno



Para poder ejecutar los .py se debe tener ultralytics instalado, para ello dos opciones:

pip install ultralytics

Crear entorno python e instalar ultralytics para linux:
```
python3 -m venv venv
source ./venv/bin/activate
pip install ultralytics
```
Para ejecutar simplemente usar ``python \*.py.`` Siendo \* el nombre del archivo a ejecutar.



Para Windows, utilizar los siguientes comandos:

```
python3 -m venv venv

.\\venv\\Scripts\\Activate.ps1

Set-ExecutionPolicy -Scope CurrentUsr -ExecutionPolicy RemoteSigned

pip install ultralytics
```



## Descripción de archivos

### caras\_extraidas
Al ejecutar el código ``pruebaModeloDeteccionCaras.py`` sobre una imagen, se guardan las caras extraídas en dicha imagen en la carpeta <kbd>caras_extraidas</kbd>

### emociones\_extraidas
Al ejecutar el código ``pruebaModeloDeteccionEmociones`` sobre una o varias caras, se guardan los resultados en la carpeta <kbd>emociones_extraidas</kbd>. Se trata de imágenes con un rectángulo delimitador de la cara y un texto codificando la emoción.

### imagenesPruebaManual
Para comprobar cómo se va comportando el modelo entrenado antes de implementar el funcionamiento en tiempo real, se prueban los modelos por separado con imágenes ejemplo que se almacenan en esta carpeta. 

### yolo\_v8 emotion
En esta carpeta se almacenan los resultados del modelo que detecta las emociones: pesos, gráficas de precisión, recall, matrices de confusión, resultados visuales de algunas épocas...


### yolov8\_face



Esta carpeta contiene los resultados del detector de caras: las matrices de confusión, las curvas de aprendizaje y comparación de parámetros con métricas. También contiene imágenes con los diferentes batches de entrenamiento con las caras detectadas, así como su grado de confianza de que lo detectado sea una cara. "*args.yaml*" muestra los datos del entrenamiento del modelo.



### aplicacionFinal.py



Ejecutable del programa final, que combina tanto la detección de caras como la detección de emociones. Para utilizarlo, una vez configurado el entorno, se debe usar la línea


```
python3 aplicacionFinal.py
```


Durante la ejecución, puede salir una ventana emergente para conceder permisos de la cámara a la aplicación.

### DeepFace.ipynb
En un principio, intentamos realizar el entrenamiento y la ejecución del proyecto usando GoogleColab, pero esto no fue posible porque sobrepásabmos los tiempos de ejecución permitidos y obteníamos una penalización. No solo eso, sino que esta plataforma no permite ejecución en tiempo real. Prueba de ello es este Notebook, en el que descargamos un modelo y podemos detectar emociones a partir de fotografías. Se plantean retos como hacer la conexión entre los recursos de Google y el hardware de nuestros dispositivos.


### emotions.yaml



Contiene tanto las rutas para determinar el conjunto de datos de entrenamiento, validación y test como las etiquetas de las emociones.



### face.yaml



Contiene tanto las rutas para determinar el conjunto de datos de entrenamiento, validación y test y la declaración de las clases (solo hay una clase posible).



### pruebaModeloDeteccionCaras.py

Script que carga el modelo entrenado de detección facial y lo ejecuta sobre imágenes de prueba ubicadas en <kbd>imagenesPruebaManual</kbd>. Recorta las regiones identificadas como rostros y las guarda individualmente en la carpeta <kbd>caras_extraidas</kbd>, permitiendo verificar la calidad de la detección de forma aislada.

### pruebaModeloDeteccionEmociones.py

Script diseñado para validar el modelo de clasificación de emociones. Toma imágenes de rostros recortados como entrada, predice la emoción correspondiente utilizando los pesos entrenados y guarda el resultado visual (imagen con etiqueta y probabilidad) en la carpeta <kbd>emociones_extraidas</kbd>.

### train_face_yolo.py

Código encargado de iniciar el entrenamiento del modelo YOLOv8 específico para la detección de caras. Lee la configuración del dataset desde ``face.yaml`` y, tras completar las épocas definidas, guarda los pesos resultantes ('best.pt' y 'last.pt') y las métricas de rendimiento en la carpeta <kbd>yolov8_face</kbd>. Destacar que como no se incluyen los datasets este archivo es simplemente para visualización de como se realizó, pues realmente su ejecución no funcionará. En caso de querer probarlo se deben descargar los datasets presentados en ``deteccion_de_emociones.pdf``.

### train_yolo_emotion.py

Código responsable del entrenamiento del modelo para la clasificación de emociones. Utiliza la configuración definida en ``emotions.yaml`` para procesar el conjunto de datos y genera los archivos de pesos, matrices de confusión y gráficas de aprendizaje que se almacenan en la carpeta <kbd>yolo_v8 emotion</kbd>. Destacar que como no se incluyen los datasets este archivo es simplemente para visualización de como se realizó, pues realmente su ejecución no funcionará. En caso de querer probarlo se deben descargar los datasets presentados en ``deteccion_de_emociones.pdf``.





