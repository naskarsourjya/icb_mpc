import do_mpc
import casadi as ca
import numpy as np
import onnx
from tensorflow import keras
from keras2casadipy import keras_to_casadi

def get_nn_model(ANN, nr_states, nr_inputs, l_NARX, is_keras=False):
    """

    This function creates a do_mpc model from a neural network model. The neural network model can be either a Keras model or an ONNX model.

    The ANN must be traind to predict from x_k, u_k -> x_k+1 if l_NARX = 1.
    If l_NARX > 1, the ANN must be trained to predict from x_k, x_k-1, ..., x_k-l_NARX+1, u_k, u_k-1, ..., u_k-l_NARX+1 -> x_k+1.
    This function is ment to work with a ANN such as the one created in example_keras_model.py.
    We define rhs as follows:

    rhs('x_k', x_next)
    rhs('x_k-1', x_k)
    rhs('x_k-2', x_k-1)
    ...

    :param onnx: ONNX model
    :param nr_states: Number of states
    :param nr_inputs: Number of inputs
    :param l_NARX: Number of past data points used for prediction
    :return: do_mpc model

    """
    model_type = 'discrete'
    model = do_mpc.model.Model(model_type, 'MX')

    # States struct (optimization variables):

    # a loop that adds states to x_k
    state_list = []
    input_list = []
    meas_list = []

    for i in range(l_NARX):
        state_list.append(model.set_variable(var_type='_x', var_name=f'x_k-{i}', shape=(nr_states, 1)))
    
    
    for i in range(l_NARX-1):    
        input_list.append(model.set_variable(var_type='_x', var_name=f'u_k-{i+1}', shape=(nr_inputs, 1)))

    
    y = ca.vertcat(*state_list)

    # Input struct (optimization variables):	
    u_k = model.set_variable(var_type='_u', var_name='u_k', shape=(nr_inputs, 1))
    
    u = ca.vertcat(u_k, ca.vertcat(*input_list))
    if is_keras:
        ANN_cas = keras_to_casadi(ANN)
        x_next = ANN_cas(y.T, u.T).reshape((nr_states, 1))
    else:
        # Use ONNX model with converter
        casadi_converter = do_mpc.sysid.ONNXConversion(ANN)
        # casadi_converter.convert(x_k = ca.SX.ones(0,15*3), u_k = ca.SX.ones(0,5*3))
        casadi_converter.convert(x_k = y.T, u_k = u.T)
        
        x_next = casadi_converter['x_next'].reshape((nr_states, 1))
    

    # Differential equations
    model.set_rhs('x_k-0', x_next)


    if l_NARX > 1:
        for i in range(l_NARX-1):
            model.set_rhs(f'x_k-{i+1}', state_list[i])

        model.set_rhs(f'u_k-{1}', u_k)
        
        for i in range(l_NARX-2):
            model.set_rhs(f'u_k-{i+2}', input_list[i])

    # Build the model
    model.setup()
    return model