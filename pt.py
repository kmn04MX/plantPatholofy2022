import numpy as np
import pandas as pd
import os
import seaborn as sns
#import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from keras.applications.inception_v3 import InceptionV3
import shutil
from shutil import copyfile
import random
import math

"""
plt.figure(figsize=(15,7))
g = sns.countplot(train['labels'], palette='icefire')
plt.title("Categoria de enfermedades")
print(train.shape)
print(train['labels'].value_counts())
"""

"""
print(len(train))
print(train.columns)
#print(train['labels'].value_counts().plot.bar())
print(train['labels'].value_counts())
"""

#print(train.sort_values(['labels'], ascending=[True, False]).head(2))
#print(train.sort_values('image', ascending=True).head(3))

##array =train['labels'].unique()
#print(array[0])
    


def clasificaImagenes():
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
    

def generaImage3():
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
    



if __name__ == "__main__":
    
    #clasificaImagenes()
    #generaImagees()
    #separarImagen()
    #graficaBarra()
    #leeCarpetas()
    #generaImage3()
    
    
    pass





