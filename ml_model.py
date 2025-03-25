import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the input and output datasets
input_data = pd.read_csv('input_dataset (1).csv')
output_data = pd.read_csv('output_dataset (1).csv')

# Preprocess the data by normalizing the input and output datasets
input_data = input_data / input_data.max()
output_data = output_data / output_data.max()

# Define a neural network model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(10000,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(420)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(input_data, output_data, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, mae = model.evaluate(input_data, output_data)
print(f'Model Loss: {loss}, Model MAE: {mae}')
