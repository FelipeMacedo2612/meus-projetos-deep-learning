#%% IMPORTS DE BIBLIOTECAS

from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

#%% IMPORTS DO DATASET

# Teste básico com o MNIST
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

#%%

train_images.shape
train_labels.shape

#%% Reshape dos inputs

X_train = train_images.reshape(-1, 28 * 28).astype('float32') / 255
X_test = test_images.reshape(-1, 28 * 28).astype('float32') / 255

#%% CRIANDO MODELO FUNCIONAL

inputs = keras.Input(shape=(784,))
features = layers.Dense(64, activation="relu")(inputs)
outputs = layers.Dense(10, activation="softmax")(features)
model = keras.Model(inputs=inputs, outputs=outputs, name="my_functional_model")

model.summary()

#%% SEPARANDO EM VAL E TRAIN

X_val, partial_x_train = X_train[:10000], X_train[10000:]
y_val, partial_y_train = train_labels[:10000], train_labels[10000:]

#%% CRIANDO MODELO

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

#%% CRIANDO CALLBACKS

callbacks = [
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr = 1e-6,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
]

#%% TREINANDO MODELO

history = model.fit(
    partial_x_train,
    partial_y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=128,
    callbacks=callbacks
)

#%%

loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, "r--", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

#%% 

accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
epochs = range(1, len(loss) + 1)
plt.plot(epochs, accuracy, "r--", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("accuracy")
plt.legend()
plt.show()