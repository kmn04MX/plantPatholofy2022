import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model
import numpy as np
import time
import matplotlib.pyplot as plt
import datetime
import cv2
import os
import pandas as pd
import seaborn as sns
import random
import math
import shutil
#import pydot
#import graphviz
 



from keras.optimizers import SGD
from keras.applications.inception_v3 import InceptionV3
os.environ['KMP_DUPLICATE_LIB_OK']='True'
    


def clasificaImagenes():
    #Esta función sirve para acomodar las imagenes en función de la clase a la que pertenecen
    df = pd.read_csv('train.csv')
    clases = df["labels"].value_counts().head(6).index.tolist()

    for clase in clases:
        os.makedirs('train_images/'+clase, exist_ok=True)
        
    
    for index, row in df.iterrows():
        imagen = row['image']
        categoria = row['labels']
        if(categoria=="scab" or categoria=="healthy" or categoria=="frog_eye_leaf_spot" or
           categoria=="rust" or categoria=="complex" or categoria=="powdery_mildew"):
            ruta_origen = 'train_images/'+imagen
            ruta_destino = 'train_images/'+categoria
            shutil.move(ruta_origen, ruta_destino)
    
        

def moverImagen():
    #Sirve para separar un porcentaje de imagenes de una carpeta a otra
    dir_origen = 'train_images/'
    dir_destino = 'test_images/'
    
    
    df = pd.read_csv('train.csv')
    clases = df["labels"].value_counts().head(6).index.tolist()
    print(clases)

    for clase in clases:
        os.makedirs(dir_destino+clase, exist_ok=True)
        nombres_archivos = os.listdir(dir_origen+clase)
        num_imagenes = len(nombres_archivos)
        num_seleccionados = int(num_imagenes*0.20)
        seleccionados = random.sample(nombres_archivos, num_seleccionados)
        for imagen in seleccionados:
            origen = os.path.join(dir_origen+clase, imagen)
            destino = os.path.join(dir_destino+clase, imagen)
            os.rename(origen,destino)


def contarContenidoCarpeta(dir_location):
    #Sirve para contar las imagenes de una carpeta
    
    
    # Especificar la ruta de la carpeta raíz
    root_folder = dir_location
    totalclases = []
    
    # Obtener una lista de todas las subcarpetas en la carpeta raíz
    subfolders = [os.path.join(root_folder, name) for name in os.listdir(root_folder)
                  if os.path.isdir(os.path.join(root_folder, name))]
    subfolders = sorted(subfolders, key=lambda x: len(os.listdir(x)), reverse=True)
    
    for subfolder in subfolders:
        tamano = len(os.listdir(subfolder))
        totalclases.append(tamano)
    return totalclases

def generaDataFrame(flag, dir_location):
    df = pd.read_csv('train.csv')
    clases = df["labels"].value_counts().head(6).index.tolist()
    totalClases = df["labels"].value_counts().head(6).unique().tolist()
    
    if(flag=="default"):
        totalClases = df["labels"].value_counts().head(6).unique().tolist()
        data = {'Categories': clases,
                'images': totalClases}
        
        df_default = pd.DataFrame(data, index=clases)
        return df_default
    else:
        totalImages = contarContenidoCarpeta(dir_location)
        data = {'Categories': clases,
            'images': totalImages}
        df_create = pd.DataFrame(data, index=clases)
        return df_create
    pass

def separarImagen():
    #En caso de que se tenga que recortar en función de la carpeta que tenga menos imagénes
    original_folder ='ztrain_images_sobrantes/'
    new_folder  = 'zprueba/'
    
    files = os.listdir(original_folder) #Arreglo con la cantidad de imagenes
    total_file = len(files) 
    num_selected = 100
    seleccionados = random.sample(files, num_selected)
    for imagen in seleccionados:
        origen = os.path.join(original_folder, imagen)
        destino = os.path.join(new_folder, imagen)
        os.rename(origen,destino)
    

def generaImage():
    # Directorio donde se encuentran las imágenes originales
    input_dir = 'validation_images/complex'
    output_dir = "validation_images/complex_al20"
    
    # Crear un generador de datos de imágenes
    datagen = ImageDataGenerator(
        rotation_range=20,  # Rotación aleatoria de 20 grados
        width_shift_range=0.10,  # Desplazamiento horizontal aleatorio de 10%
        height_shift_range=0.10,  # Desplazamiento vertical aleatorio de 10%
        shear_range=0.20,  # Cizallamiento aleatorio de 20 grados
        zoom_range=0.20,  # Zoom aleatorio de hasta 20%
        horizontal_flip=False,  # Volteo horizontal aleatorio
        vertical_flip=False,  # Volteo vertical aleatorio
        fill_mode='nearest',
        rescale=1./255,  # Normalización de píxeles
    )

    # Cargar y transformar las imágenes de un directorio
    train_generator = datagen.flow_from_directory(
        input_dir,
        target_size=(224, 224),  # Tamaño de las imágenes de salida
        batch_size=160,  # Tamaño del lote
        class_mode= None,  # Modo de clasificación
        shuffle = True,
    )
    
    # Genera 50 imágenes aumentadas para cada imagen original
    num_augmented_images = 35
    for images in train_generator:
        # Genera imágenes aumentadas para cada lote de imágenes
        for i in range(num_augmented_images):
            # Aplica transformaciones de aumento de datos a cada imagen
            augmented_images = datagen.flow(
                images,
                batch_size=1,
                save_to_dir=output_dir,
                save_prefix='augmented',
                save_format='jpg'
            )
            # Genera una imagen aumentada
            augmented_image = augmented_images.next()[0]
            
        # Rompe el bucle después de haber generado imágenes aumentadas para todas las imágenes originales
        if len(os.listdir(output_dir)) >= num_augmented_images * len(os.listdir(input_dir)):
            break

        


def graficaBarra():
    #DataFrame para datos de entrenamiento
    dir_origen = 'train_images/'
    dir_destino = 'test_images/'
    dir_validation = "validation_images/"
    
    df_default = generaDataFrame("default", "")
    df_train = generaDataFrame("", dir_origen)
    df_test = generaDataFrame("", dir_destino)
    df_validation = generaDataFrame("", dir_validation)

    
    
    fig = plt.figure(figsize=(20,10))
    fig.clf()
    ax = fig.subplots(2,2)
     
    ax[0,0].bar(df_default['Categories'],df_default['images'])
    ax[0,0].set_xlabel('Imagees')
    ax[0,0].set_ylabel('Categrias')
    ax[0,0].set_title('Total de datos')
     
    ax[0,1].bar(df_train['Categories'], df_train['images'])
    ax[0,1].set_xlabel('Imagenes')
    ax[0,1].set_ylabel('Categorias')
    ax[0,1].set_title('Datos de entrenamiento')
     
    ax[1,0].bar(df_test['Categories'], df_test['images'])
    ax[1,0].set_xlabel('Imagenes')
    ax[1,0].set_ylabel('Categorias')
    ax[1,0].set_title('Datos de prueba')
    
    
    ax[1,1].bar(df_validation['Categories'], df_validation['images'])
    ax[1,1].set_xlabel('Imagenes')
    ax[1,1].set_ylabel('Categorias')
    ax[1,1].set_title('Datos de validacion')

    fig.tight_layout()
    fig.show()
    

def showImages(image_files,class_name):
    for idx, img_path in enumerate(image_files):
        plt.subplot(3, 3, idx + 1)
        img = plt.imread(img_path)
        plt.imshow(img, cmap = 'gray')
        plt.title(class_name)
        
def plot_images(path, class_name):
    image_paths = []
    class_name_path = os.path.join(path, class_name)
    image_paths = [os.path.join(class_name_path, img_png) for img_png in random.sample(os.listdir(class_name_path), 3)]
    
    plt.figure(figsize = (10, 25))
    showImages(image_paths,class_name)

def model_neuronal_structure(model3):
    #tensorflow.keras.utils.
    plot_model(
    model3,
    to_file="model.png",
    show_shapes=True,
    show_dtype=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96,
    )
    

if __name__ == "__main__":
    
    #clasificaImagenes()
    #generaImagees()
    #separarImagen()
    #graficaBarra()
    #leeCarpetas()
    #generaImage()
    
    

    img_width, img_height = 128,128
    NAME = 'Model_CNN_{}'.format(datetime.datetime.now().strftime("%d.%m.%Y-%H_%M"))
    logdir = os.path.join("logs", NAME)
    t_board = TensorBoard(log_dir = logdir)
    train_data_dir = 'train_images/'
    validation_data_dir = 'test_images/'
    eval_data_dir = 'validation_images/'
    
    """
    nb_train_samples = 12444
    nb_validation_samples = 3107
    nb_eval_samples = 1726
    epochs = 100
    batch_size = 128
    num_of_class = 6
    classes = os.listdir(train_data_dir)"""
    
    nb_train_samples = 1322
    nb_validation_samples = 150
    nb_eval_samples = 60
    epochs = 1
    batch_size = 10
    num_of_class = 6
    classes = os.listdir(train_data_dir)
    
    
    print(f"La creación de este código es: {NAME}\n")
    
    
    graficaBarra()
    
    df_default = generaDataFrame("default", "")
    df_train = generaDataFrame("", train_data_dir)
    df_test = generaDataFrame("", validation_data_dir)
    df_validation = generaDataFrame("", eval_data_dir)
    
    print("Imagenes totales")
    print(df_default["images"])
    print("\nImmagenes para el entrenamiento")
    print(df_train["images"])
    print("\nImagenes para la pruebas")
    print(df_test["images"])
    print("\nImagenes para la validación")
    print(df_validation["images"])
    print("\n")
     
    
    
    train_datagen = ImageDataGenerator( #Image Augmentation # Research on other parameters
                    rotation_range=20,  # Rotación aleatoria de 20 grados
                    width_shift_range=0.10,  # Desplazamiento horizontal aleatorio de 10%
                    height_shift_range=0.10,  # Desplazamiento vertical aleatorio de 10%
                    shear_range=0.20,  # Cizallamiento aleatorio de 20 grados
                    zoom_range=0.20,  # Zoom aleatorio de hasta 20%
                    horizontal_flip=False,  # Volteo horizontal aleatorio
                    vertical_flip=False,  # Volteo vertical aleatorio
                    fill_mode='nearest',
                    rescale=1./255,  # Normalización de píxeles
                    )
      
    test_datagen = ImageDataGenerator(rescale = 1. / 255) #Image Augmentation
    
    
      
    train_generator = train_datagen.flow_from_directory(train_data_dir, 
                                                        target_size =(img_width,img_height), 
                                                        batch_size = batch_size, 
                                                        class_mode ='categorical',
                                                        color_mode='rgb', #grayscale
                                                        shuffle=False)
      
    validation_generator = test_datagen.flow_from_directory(validation_data_dir, 
                                                            target_size =(img_width, img_height), 
                                                            batch_size = batch_size, 
                                                            class_mode ='categorical',
                                                            color_mode='rgb', #grayscale
                                                            shuffle=False) 
    
    eval_generator = test_datagen.flow_from_directory(eval_data_dir, 
                                                      target_size =(img_width, img_height), 
                                                      batch_size = batch_size,
                                                      class_mode ='categorical',
                                                      color_mode='rgb', #grayscale
                                                      shuffle=False)
    #to show images from dataSet
    for classe in classes:
        plot_images(train_data_dir, classe)
    
    
    
    
    
    
    
    
    
    input_shape = (img_width, img_height, 3)    

    model3 = Sequential()
    
    model3.add(Conv2D(32, (3,3), input_shape=input_shape, activation='relu',data_format='channels_last'))
    model3.add(MaxPooling2D((2,2), strides=(1,1), padding='same'),)
    
    model3.add(Conv2D(64, (3,3), activation='relu'),)
    model3.add(MaxPooling2D((2,2), strides=(1,1), padding='same'),)
    
    model3.add(Conv2D(128, (3,3), activation='relu'),)
    model3.add(MaxPooling2D((2,2), strides=(1,1), padding='same'),)
    # model3.add(Dropout(0.15))
    
    model3.add(Flatten())
    
    model3.add(Dense(128, activation='relu'))
    model3.add(Dense(64, activation='relu'))
    
    model3.add(Dropout(0.5))
    model3.add(Dense(num_of_class, activation='softmax'))
    
    model3.summary()
    
    
    model_neuronal_structure(model3)
    
    
    model3.compile(loss ='categorical_crossentropy',
                     optimizer ='rmsprop', 
                   metrics =['accuracy'])
    
    H = model3.fit_generator(train_generator,
                             steps_per_epoch = nb_train_samples // batch_size, 
                             epochs = epochs, validation_data = validation_generator, 
                             validation_steps = nb_validation_samples // batch_size, callbacks = [t_board])
   
    
    
    
    
    
    
    
    
    
    
    # Plot training & validation accuracy values
    fig, ax = plt.subplots(1,1)
    plt.plot(H.history["accuracy"])
    plt.plot(H.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    
    #fig.savefig(logdir +'/' +  'model1_train_test_accuracy.jpeg' , dpi=93)
    fig.savefig('model1_train_test_accuracy.jpeg' , dpi=93)
    
    
    
    
    
    
    
    
    
    

    # Plot training & validation loss values
    fig, ax = plt.subplots(1,1)
    plt.plot(H.history['loss'])
    plt.plot(H.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    fig.savefig('model1_train_test_los.jpeg',dpi=93)
     

    
    
    
    
    

    model3.save(logdir+'/'+NAME+'.hdf5')
    
    

    validation_generator.class_indices
     


    pass