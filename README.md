# Shopping-cart

## **English version** 

The motivation of this project is to design a neural network that facilitates shopping in a supermarket. For this purpose, seven classes have been selected to be differentiated: brick of gazpacho, can of olives, can of tuna, Twirl chocolates, bag of chips (Munchitos), yourt and Coke.

### Classification

The first step carried out in the project was to design a classification neural network, FSI_project_Classification.ipynb. In this Python notebook we load the libraries and the google drive path where the images of the products are located, in this case 50 of each class. The 80% of the images are used to train the network and the remaining 20% for validation. The neural network designed has the following structure:

![image](https://user-images.githubusercontent.com/87015133/148682631-98568d38-6a13-4bc5-a873-23b31b142860.png)

The network is trained with categorical cross-entropy loss, Adams optimiser, accuracy as a metric and EarlyStopping is also used.

![image](https://user-images.githubusercontent.com/87015133/148682703-98ed83a4-a321-4a06-96fc-cfb39b3567b8.png)

An accuracy of 83% is achieved over 14 epochs. The worst classified products are those with similar colours.

#### Data augmentation

As few images are available for each class, the performance of the neural network can be improved by using data augmentation techniques. These consist of augmenting the training data set by inverting, zooming, rotating, etc. the available images to obtain new ones.

![image](https://user-images.githubusercontent.com/87015133/148682990-fd8962b5-3345-43bc-b136-d9f6dcfe35c7.png)

This results significantly improved accuracy, exceeding 95% in 13 epochs.

#### Transfer learning

Transfer learning is a widely used method for building neural network models that usually gives good results when there are not many training samples. It consists of using the convolutional layers of a previously trained network with a large number of images, freezing their weights (or features) and attaching a final fully connected part that performs the task of classifying the particular dataset. The only weights that are modified during the training phase will be those of the fully connected part.

![image](https://user-images.githubusercontent.com/87015133/148683124-20994929-4095-42ab-8494-6c00bc0394ea.png)

By having part of the network pre-trained, very high accuracy is achieved in a few epochs. The neural network learns so fast that overfitting occurs easily, to reduce this effect dropout layers can be added. 

### Location

The project can be taken a step further by making a location neural network, i.e. a network that allows us to identify different products in a single image, and thinking about the practical applications of the neural network, in a video. In this way, a mobile application could be designed where a supermarket customer takes a photo or records their shopping cart and knows how much they are going to spend. Or a camera could be installed on the conveyor belt at a supermarket and the price of the shopping could be calculated as the products pass by.

This second part of the project is based on the [object-detection-video](https://github.com/puigalex/deteccion-objetos-video) project to run object detection over a live video stream. [YOLO](https://pjreddie.com/darknet/yolo/) (**You Only Look Once**) is a model optimised to generate element detections at a very high speed, so it is a very good choice for use in video. By default this model is pre-trained to detect 80 different objects [data/coco.names](https://github.com/puigalex/deteccion-objetos-video/blob/master/data/coco.names)

#### Prepare the environment for running the code

Before running the program you can create a virtual environment and install Python, then install the libraries needed to run the Python program which are in the file requirements.txt
```
pip install -r requirements.txt
```

To run the yolo model we will have to download the neural network weights (the values of the connections between neurons), this kind of models are computationally very heavy to train from zero so downloading the pre-trained model is a good option.

```
bash weights/download_weights.sh
```

#### Running the video object detector 
You can run the program with the following command. This will start a video with the webcam that is connected to the computer and will locate streaming objects.
```
python deteccion_video.py
```

You can also run the code on an already recorded video:
```
python deteccion_video.py --webcam 0 --directorio_video <video_directory.mp4>
```

#### Customize model 

If you want to train a model with custom classes instead of using the default classes, you must first tag the images with the VOC format. 
The file ```utils/classes.names`` should contain the name of the classes, as they were labelled, one line per class. To learn more about the labelling of the images see the original project.

From the utils folder we will run the create_custom_model file to generate a .cfg file containing information about the neural network to run the detections.
```
cd utils
bash create_custom_model.sh <Number_of_classes_to_detect>
cd ..
```
The YOLO weight structure is downloaded so that transfer learning can be done on those weights.
```
cd weights
bash download_darknet.sh
cd ..
```

Train the model with the new classes:
 ```
 python train.py --model_def utils/yolov3-custom.cfg --data_config utils/custom.data --pretrained_weights weights/darknet53.conv.74 --batch_size 2
 ```

#### Running object detection on video with our classes
```
python detection_video.py --model_def utils/yolov3-custom.cfg --checkpoint_model checkpoints/yolov3_ckpt_99.pth --class_path utils/classes.names --weights_path checkpoints/yolov3_ckpt_99.pth --conf_thres 0.85 --webcam 0 --directorio_video shopping.mp4
```

The results obtained are presented as a shopping ticket in .txt format and a video in .mp4 format where the objects are located and the confidence percentage is shown, shopping_output.mp4.

![image](https://user-images.githubusercontent.com/87015133/148684936-0dcf41e8-8d68-4e7e-8278-dcc493cf206b.png)

To obtain the shopping ticket, a simple system has been designed. In each frame of the video all the objects that are visible are located, however, in order to calculate the purchase price it is necessary to know the total number of objects, not the number of times each one appears. Assuming that the camera is above a conveyor belt at a supermarket checkout, it is possible to estimate how many frames an object will be located depending on the speed of the conveyor belt and the number of frames per second captured by the camera.

The precision achieved in the video is not very high, as between 700 and 900 images should be used for each class to be distinguished, and about 70 per class have been used. However, despite the few images used, the results obtained are satisfactory and improving accuracy is a matter of spending more time labelling images.




## **Versión en español**

La motivación de este proyecto es diseñar una red neuronal que facilite la compra en un supermercado. Para ello se han seleccionado siete clases a diferenciar: brick de gazpacho, lata de aceitunas, lata de atún, chocolatinas Twirl, bolsa de papas fritas (Munchitos), yourt y Coca cola.

### Clasificación

El primer paso llevado a cabo en el proyecto fue diseñar una red neuronal de clasificación, FSI_project_Classification.ipynb. En este notebook de Python se cargan las librerías y la dirección de google drive donde se encuentran las imágenes de los productos, en este caso 50 de cada clase. El 80% de las imágenes se emplean para entrenar la red y el 20% restante para la validación. La red neuronal diseñada tiene la siguiente estructura:

![image](https://user-images.githubusercontent.com/87015133/148682631-98568d38-6a13-4bc5-a873-23b31b142860.png)

La red se entrena con pérdida de entropía cruzada categórica, optimizador Adams y en función de la precisión como métrica, y también se hace uso de la técnica EarlyStopping.

![image](https://user-images.githubusercontent.com/87015133/148682703-98ed83a4-a321-4a06-96fc-cfb39b3567b8.png)

Se consigue una precisión del 83% en 14 épocas. Los productos que se clasifican peor son aquellos que tiene colores similares.

#### Data augmentation

Como se disponen de pocas imágenes para cada clase, el rendimiento de la red neuronal se puede mejorar utilizando técnicas de data augmentation. Estas consisten en aumentar el conjunto de datos de entrenamiento invirtiendo, ampliando, rotando, etc. las imágenes disponibles para obtener otras nuevas.

![image](https://user-images.githubusercontent.com/87015133/148682990-fd8962b5-3345-43bc-b136-d9f6dcfe35c7.png)

De esta forma se obtiene una clara mejoría en la precisión, que supera el 95% en 13 épocas.

#### Transfer learning

El aprendizaje por transferencia es un método muy utilizado para construir modelos de redes neuronales que suele dar buenos resultados cuando no hay muchas muestras de entrenamiento. Consiste en utilizar las capas convolucionales de una red previamente entrenada con un gran número de imágenes, congelar sus pesos (o características) y conectar una parte final totalmente conectada que realice la tarea de clasificación del conjunto de datos concreto. Los únicos pesos que se modifican durante la fase de entrenamiento serán los de la parte totalmente conectada.

![image](https://user-images.githubusercontent.com/87015133/148683124-20994929-4095-42ab-8494-6c00bc0394ea.png)

Al tener parte de la red preentrenada, se consiguen precisiones muy altas en pocas épocas. La red neuronal aprende tan rápido que el sobreajuste se produce con facilidad, para reducir este efecto se pueden añadir capas dropout. 

### Localización

Se puede profundizar un poco más en el proyecto haciendo una red neuronal de localización, es decir, una red que nos permita identificar diferentes productos en una única imagen, y pensando en las aplicaciones prácticas de la red neuronal, en un vídeo. De esta forma, se podría diseñar una aplicación de movil donde un cliente del supermercado saque una foto o grabe su cesta de la compra y sepa cuánto se va a gastar. O bien, que se instale una cámara sobre la cinta transportadora de la caja del supermercado y se calcule el precio de la compra mientras van pasando los productos.

Esta segunda parte del proyecto está basada en el proyecto [deteccion-objetos-video](https://github.com/puigalex/deteccion-objetos-video) para correr detección de objetos sobre en un stream de video en vivo. [YOLO](https://pjreddie.com/darknet/yolo/) (**You Only Look Once**) es un modelo optimizado para generar detecciones de elementos a una velocidad muy alta, por ello es una muy buena opción para usarlo en video. Por defecto este modelo esta pre entrenado para detectar 80 objetos distintos [data/coco.names](https://github.com/puigalex/deteccion-objetos-video/blob/master/data/coco.names)

#### Preparar el entorno para ejecutar el código

Antes de ejecutar el programa se puede crear un entorno virtual e instalar Python, posteriormente se intalarán las librerías necesarias para ejecutar el programa de Python que se encuentran en el archivo requirements.txt
```
pip install -r requirements.txt
```

Para correr el modelo de yolo tendremos que descargar los pesos de la red neuronal (los valores de las conexiones entre neuronas), este tipo de modelos son computacionalmente muy pesados de entrenar desde cero por lo cual descargar el modelo pre entrenado es una buena opción.

```
bash weights/download_weights.sh
```

#### Correr el detector de objetos en video 
Se puede ejecutar el programa con el siguiente comando. Así se iniciará un video con la webcam que se encuentre conectada al ordenador y se localizarán objetos en streaming.
```
python deteccion_video.py
```

También se puede correr el código sobre un video ya grabado:
```
python deteccion_video.py --webcam 0 --directorio_video <directorio_al_video.mp4>
```

#### Personalizar modelo 

Si se quiere entrenar un modelo con clases personalizadas en vez de utilizar las clases por defecto, primero se deberán etiquetar las imagenes con el formato VOC. 
El archivo ```utils/classes.names``` debe contener el nombre de las clases, como fueron etiquetadas, un renglon por clase. Para saber más sobre el etiquetado de las imágenes consulte el proyecto original.

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

Entrenar el modelo con las clases nuevas:
 ```
 python train.py --model_def utils/yolov3-custom.cfg --data_config utils/custom.data --pretrained_weights weights/darknet53.conv.74 --batch_size 2
 ```

#### Correr deteccion de objetos en video con nuestras clases
```
python deteccion_video.py --model_def utils/yolov3-custom.cfg --checkpoint_model checkpoints/yolov3_ckpt_99.pth --class_path utils/classes.names  --weights_path checkpoints/yolov3_ckpt_99.pth  --conf_thres 0.85 --webcam 0 --directorio_video shopping.mp4
```

Los resultados obtenidos se presentan como un ticket de la compra en formato .txt y un video en formato .mp4 donde se localizan los objetos y se muestra el porcentaje de confianza, shopping_output.mp4.

![image](https://user-images.githubusercontent.com/87015133/148684936-0dcf41e8-8d68-4e7e-8278-dcc493cf206b.png)

Para obetener el ticket de la compra se ha diseñado un sistema sencillo. En cada frame del vídeo se localizan todos los objetos que sean visibles, no obstante, para poder calcular el precio de la compra es necesario saber el número total de objetos, no las veces que aparece cada uno. Suponiendo que la cámara se encuentra encima de una cinta transportadora en la caja de un supermercado, se puede estimar en cúantos frames será localizado un objeto en función de la velocidad de la cinta transportadora y el número de frames por segundo capturados por la cámara.

La precisión alcanzada en el video no es muy elevada ya que se deberían utilizar entre 700 y 900 imágenes de cada clase que se quiera distinguir y se han empleado unas 70 por clase. Sin embargo, a pesar de las pocas imágenes empleadas los resultados obtenidos son satisfactorios y mejorar la precisión es cuestión de dedicar más tiempo a etiquetar imágenes.

