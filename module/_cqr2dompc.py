import do_mpc
import torch
import casadi as ca
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import re


class Robust_Model():

    def __init__(self):
        # generating flags
        self.flags = {
            'model_ready': False
        }

    def casadi_tanh_pytorch(self, x):
        return (ca.exp(x) - ca.exp(-x)) / (ca.exp(x) + ca.exp(-x))


    def extract_numbers(self, var_name):
        numbers = re.findall(r'\d+', var_name)  # Finds all numbers in the string
        return [int(num) for num in numbers]  # Convert them to integers


    def scale_input_layer(self, layer, scaler):

        if scaler == None:
            layer_scaled = layer

        elif isinstance(scaler, MinMaxScaler):

            # extracting scaler info
            X_min = scaler.data_min_  # Minimum values of original data
            X_max = scaler.data_max_  # Maximum values of original data
            X_scale = scaler.scale_  # Scaling factor (1 / (X_max - X_min))
            X_min_target = scaler.min_  # Shift factor (used for transformation)

            # final scaling
            layer_scaled = X_min_target + X_scale * (layer - X_min)

        elif isinstance(scaler, StandardScaler):
            mean = scaler.mean_
            std = scaler.scale_

            # findal scaling
            layer_scaled = (layer - mean) / std

        else:
            raise ValueError(f"Only MinMaxScaler and StandardScaler supported as of yet! Scaler found is {scaler}, "
                             f"which is not supported.")

        # end
        return layer_scaled


    def unscale_output_layer(self, layer, scaler):

        if scaler == None:
            layer_unscaled = layer

        elif isinstance(scaler, MinMaxScaler):

            # extracting scaler info
            X_min = scaler.data_min_  # Minimum values of original data
            X_max = scaler.data_max_  # Maximum values of original data
            X_scale = scaler.scale_  # Scaling factor (1 / (X_max - X_min))
            X_min_target = scaler.min_  # Shift factor (used for transformation)

            # final scaling
            #(x_scaled - X_min_target)/X_scale + X_min
            #layer_unscaled = X_min_target + X_scale * (layer - X_min)
            layer_unscaled = (layer - X_min_target)/X_scale + X_min

        elif isinstance(scaler, StandardScaler):
            mean = scaler.mean_
            std = scaler.scale_

            # findal scaling
            layer_unscaled = layer * std + mean

        else:
            raise ValueError(f"Only MinMaxScaler and StandardScaler supported as of yet! Scaler found is {scaler}, "
                             f"which is not supported.")

        # end
        return layer_unscaled


    def nn2casadi(self, input_layer, model):

        # scaled input layer
        output_layer = self.scale_input_layer(input_layer, model.input_scaler['scaler'])

        # reading the layers and the biases
        for i, layer in enumerate(model.network):

            # linear transformations
            if isinstance(layer, torch.nn.Linear):
                # extracting weight and bias
                weight = layer.weight.cpu().detach().numpy()
                bias = layer.bias.cpu().detach().numpy()

                output_layer = ca.mtimes(weight, output_layer) + bias

            elif isinstance(layer, torch.nn.Tanh):
                output_layer = self.casadi_tanh_pytorch(output_layer)

            else:
                raise RuntimeError('{} not supported!'.format(layer))

        # scaled output layer
        output_layer_scaled = self.unscale_output_layer(output_layer, model.output_scaler['scaler'])

        return output_layer_scaled

    def convert2dompc(self, cqr, verbose = True):

        # storage
        self.cqr = cqr

        # init
        model = do_mpc.model.Model(model_type='discrete', symvar_type='SX')

        # variable setup
        d_states = {}
        d_inputs = {}
        input_layer_list = []
        d_state_list = []

        # parameter for robust mpc
        p_switch = model.set_variable(var_type='_p', var_name='p_switch', shape=(1, 1))

        # generating teh input layer
        for var_name in self.cqr.narx.x_label:
            if var_name.startswith('input') and var_name.endswith('lag_0'):
                d_inputs[var_name] = model.set_variable(var_type='_u', var_name=var_name, shape=(1, 1))
                input_layer_list.append(d_inputs[var_name])
            else:
                d_states[var_name] = model.set_variable(var_type='_x', var_name=var_name, shape=(1, 1))
                input_layer_list.append(d_states[var_name])
                d_state_list.append(var_name)

        for i, var in enumerate(input_layer_list):
            if i == 0:
                input_layer = var
            else:
                input_layer = ca.vertcat(input_layer, var)

        # extracting the functions of the nominal and 2 cqr models
        nominal_model_rhs = self.nn2casadi(input_layer=input_layer, model=self.cqr.narx.model)
        cqr_high_model_rhs = (self.nn2casadi(input_layer=input_layer, model=self.cqr.cqr_high_model)
                              + self.cqr.Q1_alpha.cpu().detach().numpy().reshape((-1, 1)))
        cqr_low_model_rhs = (self.nn2casadi(input_layer=input_layer, model=self.cqr.cqr_low_model)
                             - self.cqr.Q1_alpha.cpu().detach().numpy().reshape((-1, 1)))

        # setting up rhs
        rhs_list = []

        if verbose:
            print(f"\n\n-------- Pytorch NARX Model -> do-mpc model --------\n")
        for var_name in d_state_list:

            # init
            indices = self.extract_numbers(var_name)

            # model equation
            if var_name.startswith('state') and var_name.endswith('lag_0'):

                rhs_n = (nominal_model_rhs[indices[0] - 1, 0]
                         + ((p_switch+1)/2)* cqr_high_model_rhs[indices[0] - 1, 0]
                         + ((p_switch-1)/2)* cqr_low_model_rhs[indices[0] - 1, 0])

                rhs_list.append(rhs_n)
                model.set_rhs(var_name, rhs_n)
                if verbose:
                    print(f"{var_name} <<--- {rhs_n}")

            # state shifting
            elif var_name.startswith('state'):
                rhs_n = d_states[f'state_{indices[0]}_lag_{indices[1] - 1}']
                rhs_list.append(rhs_n)
                model.set_rhs(var_name, rhs_n)
                if verbose:
                    print(f"{var_name} <<--- {rhs_n}")

            # input
            elif var_name.startswith('input') and var_name.endswith('lag_1'):
                rhs_n = d_inputs[f'input_{indices[0]}_lag_0']
                rhs_list.append(rhs_n)
                model.set_rhs(var_name, rhs_n)
                if verbose:
                    print(f"{var_name} <<--- {rhs_n}")

            elif var_name.startswith('input'):
                rhs_n = d_states[f'input_{indices[0]}_lag_{indices[1] - 1}']
                rhs_list.append(rhs_n)
                model.set_rhs(var_name, rhs_n)
                if verbose:
                    print(f"{var_name} <<--- {rhs_n}")

        if verbose:
            print(f"\n-------- Conversion Complete. --------\n\n")

        # setting rhs
        model.setup()

        # storage
        self.model = model

        # flag update
        self.flags.update({
            'model_ready': True,
        })

        # end
        return model