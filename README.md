### Clasificación de piezas para un robot de pick and place

Actividad UA5 de Sistemas de percepción en robótica.

Un robot recoge de una cinta piezas de fijación que llegan mezcladas, arandelas, tuercas, tornillos y remaches, y las deja cada una en su caja. La cámara mira la cinta desde arriba con un panel de contraluz debajo. El sistema segmenta cada pieza con Otsu y una apertura, calcula sus características (área, circularidad, número de agujeros, excentricidad y momentos de Hu) y la clasifica con Bayes. Lo comparo con un clasificador de prototipos con los píxeles, con distancia mínima y con un perceptrón multicapa. Al final, con una imagen de la cinta con varias piezas, saco para cada una la clase, la posición, el ángulo y la caja donde la tiene que dejar el robot, y mando a revisión las que no están claras.

Las imágenes son sintéticas, las genera el script 01_generar_dataset.py y ya están guardadas en la carpeta dataset.

Para ejecutarlo hace falta un entorno de conda con las librerías del requirements.txt:

```
conda create -n vision python=3.13
conda activate vision
pip install -r requirements.txt
```

Si se quieren volver a generar las imágenes:

```
python 01_generar_dataset.py
```

Luego se abre clasificacion_piezas.ipynb en VS Code o en Jupyter con el kernel vision y se ejecutan todas las celdas con Run All. Hay que abrirlo desde esta carpeta porque las imágenes se cargan con rutas relativas.

Lo he probado con Python 3.13, NumPy 2.5.3, OpenCV 5.0.0, Matplotlib 3.11.2, Seaborn 0.13.2 y scikit-learn 1.9.1.

Resultados en test:  
-Prototipos con los píxeles: 51,5 %  
-Distancia mínima con las características: 79,0 %  
-Bayes con las características: 99,5 %  
-Perceptrón multicapa con los píxeles: 80,0 %  
-Perceptrón multicapa con las características: 98,5 %  

Archivos:  
-01_generar_dataset.py, genera el dataset y la imagen de la cinta  
-clasificacion_piezas.ipynb, entrenamiento, validación y pruebas de los clasificadores y conexión con el robot  
-dataset, 1000 imágenes de 96x96, 200 de train y 50 de test por clase, en una carpeta por clase  
-escena.png, un tramo de la cinta con nueve piezas  
-requirements.txt, las librerías necesarias
