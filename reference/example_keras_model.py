import numpy as np
import tensorflow as tf
from tensorflow import keras


def get_keras_model(states, inputs, outputs, n_layer, n_neurons):
    """
    Create and compile Keras model for system identification
    """
    data = {'x_k': states, 'u_k': inputs, 'x_next': outputs}
    
    # Model inputs
    x_k_tf = keras.Input(shape=(data['x_k'].shape[1]), name='x_k')
    u_k_tf = keras.Input(shape=(data['u_k'].shape[1]), name='u_k')
    
    # Normalization layers
    scale_input_x = keras.layers.Normalization(name='scale_input_x')
    scale_input_u = keras.layers.Normalization(name='scale_input_u')
    scale_input_x.adapt(data['x_k'])
    scale_input_u.adapt(data['u_k'])
    scale_outputs = keras.layers.Normalization(name='x_next_scaled')
    scale_outputs.adapt(data['x_next'])
    unscale_outputs = keras.layers.Normalization(invert=True, name='x_next')
    unscale_outputs.adapt(data['x_next'])

    # Scale inputs
    x_k_scaled = scale_input_x(x_k_tf)
    u_k_scaled = scale_input_u(u_k_tf)
    
    # Concatenate inputs
    layer_in = keras.layers.concatenate([x_k_scaled, u_k_scaled])
    
    # Hidden layers
    for k in range(n_layer):
        layer_in = keras.layers.Dense(
            n_neurons, 
            activation='gelu',
            name=f'hidden_{k}'
        )(layer_in)
        
        # Add dropout for regularization
        if k < n_layer - 1:  # Don't add dropout after last hidden layer
            layer_in = keras.layers.Dropout(0.1)(layer_in)
    
    # Output layer
    x_next_tf_scaled = keras.layers.Dense(data['x_next'].shape[1], name='x_next_norm')(layer_in)
    x_next_tf = unscale_outputs(x_next_tf_scaled)
    
    # Create models
    eval_model = keras.Model(inputs=[x_k_tf, u_k_tf], outputs=x_next_tf)
    train_model = keras.Model(inputs=[x_k_tf, u_k_tf], outputs=x_next_tf_scaled)
    
    return train_model, eval_model, scale_outputs