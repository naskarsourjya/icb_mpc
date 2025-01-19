import do_mpc
import casadi as ca
import numpy as np
import onnx
from tensorflow import keras


def keras_to_casadi(keras_model) -> ca.Function:
    '''
    A function that takes a Keras model as input transforms it into a casadi function.

    It can handle:
        - multiple inputs like ANN(x,k)
        - keras.layer.Normalization
        - keras.layer.Concatenate
        - keras.layer.Dense
            |_ keras.activations.tanh
            |_ keras.activations.sigmoid
            |_ keras.activations.relu
            |_ keras.activations.linear

    '''
    # Number of inputs to the ANN
    _nr_inputs = len(keras_model.inputs)
    # create a dict with keys for every layer name 
    input_symbols = [ca.MX.sym(keras_model.inputs[i].name, 1,keras.backend.int_shape(tensor)[1]) for i, tensor in enumerate(keras_model.inputs)]
    input_symbol_dict = {symbol.name(): symbol for symbol in input_symbols}

    # epsilon is added to variance to avoid devision by zero 
    epsilon = keras.backend.epsilon()


    # Map layer formulations to the corresponding output name stored in input_symbol_dict
    for layer in keras_model.layers:
        
        # Handle normalization layer
        if isinstance(layer, keras.layers.Normalization):
            if layer.invert:
                input_symbol_dict[layer.name] = (input_symbol_dict[layer.input.name.split('/')[0]]*ca.fmax(ca.sqrt((ca.DM(layer.variance.numpy()) )), ca.DM(epsilon)) + ca.DM(layer.mean.numpy()))
            else:
                input_symbol_dict[layer.name] = (input_symbol_dict[layer.input.name.split('/')[0]] - ca.DM(layer.mean.numpy()))/ca.fmax(ca.sqrt((ca.DM(layer.variance.numpy()) )), ca.DM(epsilon))

        # Handle Concatination layer   
        if isinstance(layer, keras.layers.Concatenate):
            _concat_inputs = [input_symbol_dict[i.name.split('/')[0]] for i in layer.input]
            input_symbol_dict[layer.name] = ca.horzcat(*_concat_inputs)
        # Handle Dense layer   
        if isinstance(layer, keras.layers.Dense):
            _dense_input = input_symbol_dict[layer.input.name.split('/')[0]]
            _dense_linear = ca.mtimes(_dense_input, ca.DM(layer.kernel.numpy())) + ca.DM(layer.bias.numpy().reshape(1,-1))
            
            # Add activation to Dense layer
            if layer.activation == keras.activations.tanh:
                input_symbol_dict[layer.name] = ca.tanh(_dense_linear)
            if layer.activation == keras.activations.sigmoid:
                input_symbol_dict[layer.name] = 1 / (1 + ca.exp(-_dense_linear))
            if layer.activation == keras.activations.relu:
                input_symbol_dict[layer.name] = ca.fmax(_dense_linear, 0)
            if layer.activation == keras.activations.linear:
                input_symbol_dict[layer.name] = _dense_linear

    # Get final formulation of the last layer
    _x_next = input_symbol_dict[keras_model.layers[-1].name]

    # Define casadi Function
    casadi_fn = ca.Function('ANN',input_symbols[0:_nr_inputs] , [_x_next])

    return casadi_fn