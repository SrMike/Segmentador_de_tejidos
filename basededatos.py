# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 18:05:58 2021

@author: Miguel
"""

import os
from PIL import Image
from torch.utils.data import Dataset, dataset
import numpy as np
import nibabel as nib
from tqdm import tqdm, notebook
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
#========Clase de entrenamiento para la Red Neuronal============================
#=Debe actualizarse LITS() si se quire usar de nuevo. ==========================
#LITS presenta errores al usar 2 archivos ======================================
class datas_train(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, class_dimention = False):
      self.image_dir = image_dir # directorio de los volumenes en formato string
      self.mask_dir = mask_dir   # directorio de las mascaras en formato string
      self.transform = transform
      self.class_dimention = class_dimention
      self.index_array = np.zeros([0,0])  

      if 'index_array' in os.listdir(image_dir):
        print('Cargando indices... ')
        self.index_array = np.loadtxt(image_dir + '/index_array') # carga el indice de datos generado previamente
        list_image_dir = os.listdir(image_dir)
        list_image_dir.remove('index_array')
        self.images = self.ordena_lista(list_image_dir) # lista con los nombres de cada uno
                                                        # de los archivos #volumen-xx.nii
        print('\b Listo! ')
      else:
        
        self.images = self.ordena_lista(os.listdir(image_dir))  # lista con los nombres de cada uno
                                                                # de los archivos #volumen-xx.nii

      self.mask = self.ordena_lista(os.listdir(mask_dir))    # lista con ['mascaras-00.nii',...]
      #print('images list')
      #print(os.listdir(image_dir))
      #print(self.images)
      #print('mask list')
      #print(os.listdir(mask_dir))
      #print(self.mask)
      
      #====En esta sección se generan las listas que contienen los archivos .nii
      self.list_images = [] # son los volumenes
      self.list_mask = []   # son los segmentos
      self.tamaños = []     # Lista que contiene el número de frames por 
                            # archivo
      
      for i in range(len(self.images)): 
        #--- se cargan los archivos " .nii" utilizando nib.load()
        imag = nib.load(self.image_dir + '/' + self.images[i])
        mask = nib.load(self.mask_dir + '/' + self.mask[i])
        #_se guardan en las listas
        self.list_images.append(imag)
        self.list_mask.append(mask)
        #_se consulta el tamaño y se guarda en la lista tamaños
        self.tamaños.append(imag.shape[2])
      #self.total = sum(self.tamaños)
      #=========================================================================
      #==================En esta sección se genera self.index_array si no existe
      if self.index_array.shape == (0,0):
        print('Generando indices... ')
        self.index_array = np.zeros([sum(self.tamaños),2])
        self.cont = 0

        for i in notebook.tqdm(range(len(self.tamaños)), desc= '=> Cargando base de datos', leave = False):
          for j in notebook.tqdm(range(self.tamaños[i]), desc = '=> '+self.images[i], leave = False):
            if self.list_mask[i].slicer[:,:,j:j+1].get_fdata().sum() != 0:
              self.index_array[self.cont,0] = i
              self.index_array[self.cont,1] = j
              self.cont = self.cont + 1
        self.index_array = self.index_array[0:self.cont,:]

        np.savetxt(self.image_dir + '/index_array',self.index_array)
        #print(self.cont, self.index_array.shape)
        #print(self.index_array[self.cont-1,:])
        print('\b Listo!')
        

    def __len__(self):
        return self.index_array.shape[0]-1

    def __getitem__(self, index):
      #print(index)
      if (index == 0):
        index = 1
      if (index == 64):
        index = 63
      if (index == 238):
        index = 237
        
      nlist,idx = np.int16(self.index_array[index])
      image = self.list_images[nlist].slicer[:,:,idx-1:idx+2].get_fdata()
      mask = self.list_mask[nlist].slicer[:,:,idx:idx+1].get_fdata()
      #image = image + np.abs(image.min())
      #image = (image/image.max())*255
      #image = np.uint8(image)
      #image = image[:,:,0]
      #mask = mask + 1
      if self.class_dimention:

        mat = np.zeros((mask.shape[0], mask.shape[1],2))
        mat[:,:,0] = mask[:,:,0] == 1
        mat[:,:,1] = mask[:,:,0] == 2
      
      else:
        mat = np.zeros((mask.shape[0], mask.shape[1]))
        mat[:,:] = mask[:,:,0]
        
      mask = mat
        #img_path = os.path.join(self.image_dir, self.images[index])
        #mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        #image = np.array(Image.open(img_path).convert("RGB"))
        #mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        #mask[mask == 255.0] = 1.0

      if self.transform is not None:
          
          augmentations = self.transform(image=image, mask=mask)
          image = augmentations["image"]
          mask = augmentations["mask"]

          if self.class_dimention:

            mat = torch.zeros((2,mask.shape[0], mask.shape[1]), dtype = torch.long)
            mat[0,:,:] = mask[:,:,0]
            mat[1,:,:] = mask[:,:,1]

          else:

            mat = torch.zeros((mask.shape[0], mask.shape[1]), dtype = torch.long)
            
            mat[:,:] = mask[:,:].long()

          mask = mat
      return image, mask
    def obtener_numero(self,nombre):
      a = nombre.find('-')
      if nombre[a+2]=='.':
        return int(nombre[a+1])
      elif nombre[a+3] == '.':
        return int(nombre[a+1]+nombre[a+2])
      else:
        return int(nombre[a+1]+nombre[a+2]+nombre[a+3])

    def ordena_lista(self, lista):
      # Esta función ordena de menor a mayor los nombres en formato string 
      # contenidos en la lista de entrada
      dic = {}
      print(lista)
      lista.remove('.ipynb_checkpoints')
      #print(lista)
      for i in lista:
        n = self.obtener_numero(i)
        #print(n, i)
        dic[n] = i
      lista = []
      for i in range(min(dic),max(dic)+1):
        lista.append(dic[i])
      return lista
#=======Clase data===============================================
#=Se utiliza en la gui para cargar el self.volumen
#= recibe 'datos = info_{dicom,nibabel}(/ruta/)[0]'
#=no contiene mascaras
T = A.Compose(
        [
            A.Resize(height=448, width=448),
            #A.Rotate(limit=35, p=1.0),
            #A.HorizontalFlip(p=0.5),
            #A.VerticalFlip(p=0.1),
            #A.Normalize(
            #    mean=[0.0],  # modificar para que sean mas canales de entrada
            #    std=[1.0],    # modificar para que sean mas canales de entrada
            #    max_pixel_value=255.0,
            #),
            ToTensorV2(),
        ],
    )
class datas(Dataset):
  def __init__(self,datos, transform = T):
    self.datos = datos
  def __len__(self):
    return self.datos.shape[2]
  def __getitem__(self,index):
    return self.datos[:,:,index]
  def getitem(self,index):
    #print("tamaño: ", self.datos.shape)
    rf = np.zeros([3,self.datos.shape[0],self.datos.shape[1]])
    ri = self.datos[:,:,index-1:index+2]
    rf[0,:,:] = ri[:,:,0]
    rf[1,:,:] = ri[:,:,1]
    rf[2,:,:] = ri[:,:,2]
    r = torch.from_numpy(rf).float().unsqueeze_(0)
    #return r
    if self.transform is not None:
          
          augmentations = self.transform(image = r)#image=image, mask=mask)
          r = augmentations["image"]
          #mask = augmentations["mask"]
    return r

#===============================================================
class LiTS(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, class_dimention = False):
      self.image_dir = image_dir # directorio de los volumenes en formato string
      self.mask_dir = mask_dir   # directorio de las mascaras en formato string
      self.transform = transform
      self.class_dimention = class_dimention
      self.index_array = np.zeros([0,0])  

      if 'index_array' in os.listdir(image_dir):
        print('Cargando indices... ')
        self.index_array = np.loadtxt(image_dir + '/index_array') # carga el indice de datos generado previamente
        list_image_dir = os.listdir(image_dir)
        list_image_dir.remove('index_array')
        self.images = self.ordena_lista(list_image_dir) # lista con los nombres de cada uno
                                                        # de los archivos #volumen-xx.nii
        print('\b Listo! ')
      else:

        self.images = self.ordena_lista(os.listdir(image_dir))  # lista con los nombres de cada uno
                                                                # de los archivos #volumen-xx.nii

      self.mask = self.ordena_lista(os.listdir(mask_dir))    # lista con ['mascaras-00.nii',...]
      
      
      #====En esta sección se generan las listas que contienen los archivos .nii
      self.list_images = [] # son los volumenes
      self.list_mask = []   # son los segmentos
      self.tamaños = []     # Lista que contiene el número de frames por 
                            # archivo
      
      for i in range(len(self.images)-1): 
        #--- se cargan los archivos " .nii" utilizando nib.load()
        imag = nib.load(self.image_dir + '/' + self.images[i])
        mask = nib.load(self.mask_dir + '/' + self.mask[i])
        #_se guardan en las listas
        self.list_images.append(imag)
        self.list_mask.append(mask)
        #_se consulta el tamaño y se guarda en la lista tamaños
        self.tamaños.append(imag.shape[2])
      #self.total = sum(self.tamaños)
      #=========================================================================
      #==================En esta sección se genera self.index_array si no existe
      if self.index_array.shape == (0,0):
        print('Generando indices... ')
        self.index_array = np.zeros([sum(self.tamaños),2])
        self.cont = 0

        for i in notebook.tqdm(range(len(self.tamaños)), desc= '=> Cargando base de datos', leave = False):
          for j in notebook.tqdm(range(self.tamaños[i]), desc = '=> '+self.images[i], leave = False):
            if self.list_mask[i].slicer[:,:,j:j+1].get_fdata().sum() != 0:
              self.index_array[self.cont,0] = i
              self.index_array[self.cont,1] = j
              self.cont = self.cont + 1
        self.index_array = self.index_array[0:self.cont,:]
        np.savetxt(self.image_dir + '/index_array',self.index_array)
        #print(self.cont, self.index_array.shape)
        #print(self.index_array[self.cont-1,:])
        print('\b Listo!')
        

    def __len__(self):
        return self.index_array.shape[0]

    def __getitem__(self, index):
      nlist,idx = np.int16(self.index_array[index])
      image = self.list_images[nlist].slicer[:,:,idx-1:idx+2].get_fdata()
      mask = self.list_mask[nlist].slicer[:,:,idx:idx+1].get_fdata()
      #image = image + np.abs(image.min())
      #image = (image/image.max())*255
      #image = np.uint8(image)
      #image = image[:,:,0]

      if self.class_dimention:

        mat = np.zeros((mask.shape[0], mask.shape[1],2))
        mat[:,:,0] = mask[:,:,0] == 1
        mat[:,:,1] = mask[:,:,0] == 2
      
      else:
        mat = np.zeros((mask.shape[0], mask.shape[1]))
        mat[:,:] = mask[:,:,0]
        
      mask = mat
        #img_path = os.path.join(self.image_dir, self.images[index])
        #mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        #image = np.array(Image.open(img_path).convert("RGB"))
        #mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        #mask[mask == 255.0] = 1.0

      if self.transform is not None:
          
          augmentations = self.transform(image=image, mask=mask)
          image = augmentations["image"]
          mask = augmentations["mask"]

          if self.class_dimention:

            mat = torch.zeros((2,mask.shape[0], mask.shape[1]), dtype = torch.long)
            mat[0,:,:] = mask[:,:,0]
            mat[1,:,:] = mask[:,:,1]

          else:

            mat = torch.zeros((mask.shape[0], mask.shape[1]), dtype = torch.long)
            
            mat[:,:] = mask[:,:].long()

          mask = mat
      return image, mask
    def obtener_numero(self,nombre):
      a = nombre.find('-')
      if nombre[a+2]=='.':
        return int(nombre[a+1])
      elif nombre[a+3] == '.':
        return int(nombre[a+1]+nombre[a+2])
      else:
        return int(nombre[a+1]+nombre[a+2]+nombre[a+3])

    def ordena_lista(self,lista):
      # Esta función ordena de menor a mayor los nombres en formato string 
      # contenidos en la lista de entrada
      dic = {}
      for i in lista:
        n = self.obtener_numero(i)
        dic[n] = i
      lista = []
      for i in range(min(dic),max(dic)):
        lista.append(dic[i])
      return lista
#Coment
class nuevo_tejido(Dataset):
  def __init__(self, arreglo):
    self.datos = arreglo
  def __len__(self):
    return self.index_array.shape[0]
  def __getitem__(self, index):
    return