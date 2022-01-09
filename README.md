# Shopping-cart

## **Versión en español**

La motivación de este proyecto es diseñar una red neuronal que facilite la compra en un supermercado. Para ello se han seleccionado cuatro clases a diferenciar: brick de gazpacho, lata de aceitunas, lata de atún, chocolatinas Twirl, bolsa de papas fritas (Munchitos), yourt y Coca cola.

### Clasificación

El primer paso llevado a cabo en el proyecto fue diseñar una red de clasificación, FSI_project_Classification.ipynb. En este notebook de Python se cargan las librerías y la dirección de google drive donde se encuentran las imágenes de los productos, en este caso 50 de cada clase. El 80% de las imágenes se emplean para entrenar la red y el 20% restante para la validación. La red neuronal diseñada tiene la siguiente estructura:

![image](https://user-images.githubusercontent.com/87015133/148682631-98568d38-6a13-4bc5-a873-23b31b142860.png)

La red se entrena con pérdida de entropía cruzada categórica, optimizador Adams y en función de la precisión como métrica, y también se hace uso de la técnica EarlyStopping.

![image](https://user-images.githubusercontent.com/87015133/148682703-98ed83a4-a321-4a06-96fc-cfb39b3567b8.png)

Se consigue una precisión del 83% en 14 épocas. Los productos que se clasifican peor son aquellos que tiene colores similares.

##### Data augmentation

Como se disponen de pocas imágenes para cada clase, el rendimiento de la red neuronal se puede mejorar utilizando técnicas de data augmentation. Estas consisten en aumentar el conjunto de datos de entrenamiento invirtiendo, ampliando, rotando, etc. las imágenes disponibles para obtener otras nuevas.

![image](https://user-images.githubusercontent.com/87015133/148682990-fd8962b5-3345-43bc-b136-d9f6dcfe35c7.png)

De esta forma se obtiene una clara mejoría en la precisión, que supera el 95% en 13 épocas.

##### Transfer learning

El aprendizaje por transferencia es un método muy utilizado para construir modelos de redes neuronales que suele dar buenos resultados cuando no hay muchas muestras de entrenamiento. Consiste en utilizar las capas convolucionales de una red previamente entrenada con un gran número de imágenes, congelar sus pesos (o características) y conectar una parte final totalmente conectada que realice la tarea de clasificación del conjunto de datos concreto. Los únicos pesos que se modifican durante la fase de entrenamiento serán los de la parte totalmente conectada.

![image](https://user-images.githubusercontent.com/87015133/148683124-20994929-4095-42ab-8494-6c00bc0394ea.png)

Al tener parte de la red preentrenada, se consiguen precisiones muy altas en pocas épocas. La red neuronal aprende tan rápido que el sobreajuste se produce con facilidad, para reducir este efecto se pueden añadir capas dropout. 

### Localización

Se puede profundizar un poco más en el proyecto haciendo una red de localización, es decir, una red que nos permita identificar diferentes productos en una única imagen, y pensando en las aplicaciones prácticas de la red neuronal, en un vídeo. De esta forma, se podría diseñar una aplicación de movil donde un cliente del supermercado saque una foto o grabe su cesta de la compra y sepa cuánto se va a gastar. O bien que se instale una cámara sobre la cinta transportadora de la caja del supermercado y calcule el precio de la compra.

Esta segunda parte del proyecto está basada en el proyecto [deteccion-objetos-video](https://github.com/puigalex/deteccion-objetos-video) para correr detección de objetos sobre en un stream de video en vivo. [YOLO](https://pjreddie.com/darknet/yolo/) (**You Only Look Once**) es un modelo optimizado para generar detecciones de elementos a una velocidad muy alta, por ello es una muy buena opción para usarlo en video. Por defecto este modelo esta pre entrenado para detectar 80 objetos distintos [data/coco.names](https://github.com/puigalex/deteccion-objetos-video/blob/master/data/coco.names)

##### Preparar el entorno para ejecutar el código

Antes de ejecutar el programa se puede crear un entorno virtual e instalar Python, posteriormente se intalarán las librerías necesarias para ejecutar el programa de Python que se encuentran en el archivo requirements.txt
```
pip install -r requirements.txt
```

Para correr el modelo de yolo tendremos que descargar los pesos de la red neuronal (los valores de las conexiones entre neuronas), este tipo de modelos son computacionalmente muy pesados de entrenar desde cero por lo cual descargar el modelo pre entrenado es una buena opción.

```
bash weights/download_weights.sh
```

##### Correr el detector de objetos en video 
Se puede ejecutar el programa con el siguiente comando. Así se iniciará un video con la webcam que se encuentre conectada al ordenador y se localizarán objetos en streaming.
```
python deteccion_video.py
```

También se puede correr el código sobre un video ya grabado:
```
python deteccion_video.py --webcam 0 --directorio_video <directorio_al_video.mp4>
```

##### Personalizar modelo 

Si se quiere entrenar un modelo con clases personalizadas en vez de utilizar las clases por defecto, primero se deberán etiquetar las imagenes con el formato VOC.

Desde la carpeta utils correremos el archivo create_custom_model para generar un archivo .cfg que contiene información sobre la red neuronal para correr las detecciones
```
cd utils
bash create_custom_model.sh <Numero_de_clases_a_detectar>
cd ..
```
Se descarga la estructura de pesos de YOLO para poder hacer transfer learning sobre esos pesos
```
cd weights
bash download_darknet.sh
cd ..
```

Las imagenes etiquetadas tienen que estar en el directorio **data/custom/images** mientras que las etiquetas/metadata de las imagenes tienen que estar en **data/custom/labels**.
Por cada imagen .jpg debe de existir un imagen.txt (metadata con el mismo nombre de la imagen). El archivo ```data/custom/classes.names``` debe contener el nombre de las clases, como fueron etiquetadas, un renglon por clase.

Entrenar el modelo con las clases nuevas:
 ```
 python train.py --model_def utils/yolov3-custom.cfg --data_config utils/custom.data --pretrained_weights weights/darknet53.conv.74 --batch_size 2
 ```

##### Correr deteccion de objetos en video con nuestras clases
```
python deteccion_video.py --model_def utils/yolov3-custom.cfg --checkpoint_model checkpoints/yolov3_ckpt_99.pth --class_path utils/classes.names  --weights_path checkpoints/yolov3_ckpt_99.pth  --conf_thres 0.85 --webcam 0 --directorio_video shopping.mp4
```

Los resultados obtenidos se presentan como un ticket de la compra en formato .txt y un video en formato .mp4 donde se localizan los objetos y se muestra el porcentaje de confianza, shopping_output.mp4.

![image](https://user-images.githubusercontent.com/87015133/148684323-a68e6668-749e-45e6-8c56-5e035fafcc3e.png)

Para obetener el ticket de la compra se ha diseñado un sistema sencillo. En cada frame del vídeo se localizan todos los objetos que sean visibles, no obstante, para poder calcular el precio de la compra es necesario saber el número total de objetos, no las veces que aparece cada uno. Suponiendo que la cámara se encuentra encima de una cinta transportadora en la caja de un supermercado, se puede estimar en cúantos frames será localizado un objeto en función de la velocidad de la cinta transportadora y el número de frames por segundo capturados por la cámara.

La precisión alcanzada en el video no es muy elevada ya que se deberían utilizar entre 700 y 900 imágenes de cada clase que se quiera distinguir y se han empleado unas 70 por clase. Sin embargo, a pesar de las pocas imágenes empleadas los resultados obtenidos son satisfactorios y mejorar la precisión es cuestión de dedicar más tiempo a etiquetar imágenes.

