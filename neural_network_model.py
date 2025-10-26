import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Define a simple neural network model
def create_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(20,)),  # Input layer
        Dense(32, activation='relu'),                     # Hidden layer
        Dense(1, activation='sigmoid')                   # Output layer
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Create the model
model = create_model()

# Save the model in .h5 format for Netron visualization
model.save('neural_network_model.h5')

print("Model created and saved as 'neural_network_model.h5'.")