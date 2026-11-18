#%% 
 
# CLASSIFICAÇÃO DE DIGITOS COM MODELOS BÁSICOS DE REDE NEURAL E MODELO SEQUECIAL (DANDO RESHAPE NOS INPUTS)

#%%

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt

#%%

import torchvision

train_dataset = torchvision.datasets.SVHN(root='./data', split='train', download=True)
test_dataset = torchvision.datasets.SVHN(root='./data', split='test', download=True)

X_train, y_train = train_dataset.data, train_dataset.labels
X_test, y_test = test_dataset.data, test_dataset.labels

#%%
# Ajusta o formato de (N, C, H, W) para (N, H, W, C) padrão do Keras
X_train = X_train.transpose(0, 2, 3, 1)
X_test = X_test.transpose(0, 2, 3, 1)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

#%% 

print("X_train shape:", y_train.shape)
print("X_test shape:", y_test.shape)

#%%

X_train_flat = X_train.reshape(-1, 32 * 32 * 3)
X_test_flat = X_test.reshape(-1, 32 * 32 * 3)

#%% SEPARANDO OS DADOS EM VAL E TRAIN

x_val = X_train_flat[:10000]
partial_x_train = X_train_flat[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#%% CRIANDO O MODELO

def get_model():
  model = keras.Sequential(
      [
          layers.Dense(512, activation="relu"),
          layers.Dense(256, activation="relu"),
          layers.Dense(128, activation="relu"),
          layers.Dense(10, activation="softmax")
      ]
  )

  model.compile(
      optimizer="adam",
      loss="sparse_categorical_crossentropy",
      metrics=["accuracy"]
  )

  return model


#%% TREINANDO

modelo = get_model()
history = modelo.fit(
    partial_x_train,
    partial_y_train,
    validation_data = (x_val, y_val),
    epochs=10,
    batch_size=64,
    verbose = 0
)

#%% PLOTANDO A LOSS EM UM GRÁFICO

loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, "r--", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.xticks(epochs)
plt.ylabel("Loss")
plt.legend()
plt.show()

#%% PLOTANDO A ACURÁCIA EM UM GRÁFICO

plt.clf()
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
plt.plot(epochs, acc, "r--", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.xticks(epochs)
plt.ylabel("Accuracy")
plt.legend()
plt.show()