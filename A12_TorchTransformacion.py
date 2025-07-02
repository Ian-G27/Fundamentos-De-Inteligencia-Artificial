#=================================
# Transformaciones de Tensores
#=================================
# Ian Diego Buendia Alvarez
# Fundamentos de IA
# ESFM IPN Abril 2025
#=================================

#========================
# Módulos necesarios
#=======================
import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np

#==================================
# Clase WineDataset hija de Dataset
#===================================
class WineDataset(Dataset):

  #========================
  # Constructor
  #=======================
  def __init__(self, transform=None):
    xy = np.loadtxt('C:/Users/ian_d/OneDrive/Documentos/Fundamentos-De-Inteligencia-Artificial/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
    self.n_samples = xy.shape[0]

    # note que no convertimos en tensor aquí
    self.x_data = xy[:, 1:]
    self.y_data = xy[:, [0]]

    self.transform = transform

  #==========================
  # Método para obtener datos
  #==========================
  def __getitem__(self, index):
    sample = self.x_data[index], self.y_data[index]

    if self.transform:
        sample = self.transform(sample)
    return sample
  #=============================
  # Tamaño del conjunto de datos
  #==============================
  def __len__(self):
    return self.n_samples

#==============================
# Transformaciones conumes
#=============================

#=============================
# De numpy a tensor pytorch
#=======================
class ToTensor:
  def __call__(self, sample):
    inputs, targets = sample
    return torch.from_numpy(inputs), torch.from_numpy(targets)

#===============================================
# Escalar datos (multoplicarlos por una constante)
#================================================
class MulTransform:
  def __init__(self, factor):
    self.factor = factor
  def __call__(self, sample):
    inputs, targets = sample
    inputs *= self.factor
    return inputs, targets

#=======================
# Programa principal
#====================
if __name__ == "__main__":

  print('Sin transformación')
  dataset = WineDataset()
  first_data = dataset[0]
  features, labels = first_data
  print( type(features), type(labels))
  print(features, labels)

  print('\nTransformado en tensor')
  dataset = WineDataset(transform=ToTensor())
  first_data = dataset[0]
  features, labels = first_data
  print(type(features), type(labels))
  print(features, labels)

  print('\n Con transformación a tensor y multiplicación')
  composed = torchvision.transforms.Compose([ToTensor(), MulTransform(4)])
  dataset = WineDataset(transform=composed)
  first_data = dataset[0]
  features, labels = first_data
  print(type(features), type(labels))
  print(features, labels)