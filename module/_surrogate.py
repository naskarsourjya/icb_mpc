import torch
import do_mpc
import casadi as ca
import  numpy as np
import pandas as pd
import re
from numpy.core.defchararray import index
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Surrogate():
    def __init__(self, n_x, n_u, order, t_step, debug= True):

        self.n_x = n_x
        self.n_u = n_u
        self.order = order
        self.t_step = t_step
        self.history = None
        self.log = None

        # generating flags
        self.flags = {
            'model_ready': False,
            'simulator_ready': False,
            'initial_condition_ready': False,
            'debug': debug
        }

        pass

    def casadi_tanh_pytorch(self, x):
        return (ca.exp(x) - ca.exp(-x)) / (ca.exp(x) + ca.exp(-x))


    def narx_2_dompc_model(self, narx, verbose = True):

        # init
        model = do_mpc.model.Model(model_type='discrete', symvar_type='SX')

        # variable setup
        d_states = {}
        d_inputs = {}
        input_layer_list = []
        d_state_list = []

        for var_name in narx.x_label:
            if var_name.startswith('input') and var_name.endswith('lag_0'):
                d_inputs[var_name] = model.set_variable(var_type='_u', var_name=var_name, shape=(1, 1))
                input_layer_list.append(d_inputs[var_name])
            else:
                d_states[var_name] = model.set_variable(var_type='_x', var_name=var_name, shape=(1, 1))
                input_layer_list.append(d_states[var_name])
                d_state_list.append(var_name)

        for i, var in enumerate(input_layer_list):
            if i==0:
                input_layer = var
            else:
                input_layer = ca.vertcat(input_layer, var)

        # used by random state tracking algo
        state_ref = model.set_variable(var_type='_tvp', var_name='state_ref', shape=(self.n_x, 1))
        #ref_x = model.set_variable(var_type='_x', var_name='ref_x', shape=(self.n_x, 1))
        #ref_u = model.set_variable(var_type='_u', var_name='ref_u', shape=(self.n_x, 1))

        # scaled input layer
        input_layer_scaled = self.scale_input_layer(input_layer, narx.scaler)

        # reading the layers and the biases
        for i, layer in enumerate(narx.model.network):

            # linear transformations
            if isinstance(layer, torch.nn.Linear):
                # extracting weight and bias
                weight = layer.weight.cpu().detach().numpy()
                bias = layer.bias.cpu().detach().numpy()

                if i == 0:
                    output_layer = ca.mtimes(weight, input_layer_scaled) + bias

                else:
                    output_layer = ca.mtimes(weight, output_layer) + bias

            elif isinstance(layer, torch.nn.Tanh):
                if i == 0:
                    #output_layer = ca.tanh(input_layer_scaled)
                    output_layer = self.casadi_tanh_pytorch(input_layer_scaled)

                else:
                    #output_layer = ca.tanh(output_layer)
                    output_layer = self.casadi_tanh_pytorch(output_layer)

            else:
                raise RuntimeError('{} not supported!'.format(layer))

        # setting up rhs
        rhs_list = []

        if verbose:
            print(f"\n\n-------- Pytorch NARX Model -> do-mpc model --------\n")
        for var_name in d_state_list:

            # init
            indices = self.extract_numbers(var_name)

            # model equation
            if var_name.startswith('state') and var_name.endswith('lag_0'):
                rhs_n = output_layer[indices[0]-1, 0]
                rhs_list.append(rhs_n)
                model.set_rhs(var_name, rhs_n)
                if verbose:
                    print(f"{var_name} <<--- {rhs_n}")


            # state shifting
            elif var_name.startswith('state'):
                rhs_n = d_states[f'state_{indices[0]}_lag_{indices[1]-1}']
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
        #model.set_rhs(ref_x, ref_u)
        model.setup()

        # storage
        self.model = model

        # flag update
        self.flags.update({
            'model_ready': True,
        })

        # end
        return model

    def extract_numbers(self, var_name):
        numbers = re.findall(r'\d+', var_name)  # Finds all numbers in the string
        return [int(num) for num in numbers]  # Convert them to integers


    def scale_input_layer(self, input_layer, scaler):

        if scaler == None:
            input_layer_scaled = input_layer

        elif isinstance(scaler, MinMaxScaler):

            # extracting scaler info
            X_min = scaler.data_min_  # Minimum values of original data
            X_max = scaler.data_max_  # Maximum values of original data
            X_scale = scaler.scale_  # Scaling factor (1 / (X_max - X_min))
            X_min_target = scaler.min_  # Shift factor (used for transformation)

            # final scaling
            input_layer_scaled = X_min_target + X_scale * (input_layer - X_min)

        elif isinstance(scaler, StandardScaler):
            mean = scaler.mean_
            std = scaler.scale_

            # findal scaling
            input_layer_scaled = (input_layer - mean) / std

        else:
            raise ValueError(f"Only MinMaxScaler and StandardScaler supported as of yet! Scaler found is {scaler}, "
                             f"which is not supported.")

        # end
        return input_layer_scaled

    def create_simulator(self):
        assert self.flags['model_ready'], "do_mpc model not generated! Use narx_2_dompc_model() to generate a model."

        # init
        simulator = do_mpc.simulator.Simulator(model=self.model)
        simulator.set_param(t_step=self.t_step)
        tvp_template = simulator.get_tvp_template()

        def tvp_fun(t_ind):
            return tvp_template

        simulator.set_tvp_fun(tvp_fun)
        simulator.setup()

        # storage
        self.simulator = simulator

        # flag update
        self.flags.update({
            'simulator_ready': True,
        })

        return None


    def reshape(self, array, shape):

        # rows and columns
        rows, cols = shape

        # end
        return array.reshape(cols, rows).T


    @property
    def states(self):
        return self._states


    @states.setter
    def states(self, val):
        assert isinstance(val, np.ndarray), "states must be a numpy.array."

        assert val.shape[0] == self.order, \
            'Number of samples must be equal to the order of the NARX model!'

        assert val.shape[1] == self.n_x, (
            'Expected number of states is: {}, but found {}'.format(self.n_x, val.shape[0]))

        # storage
        self._states = val


    @property
    def inputs(self):
        return self._inputs


    @inputs.setter
    def inputs(self, val):
        if self.order > 1:
            assert isinstance(val, np.ndarray), "inputs must be a numpy.array."

            assert self.order - 1 == val.shape[0], \
                'Number of samples for inputs should be (order-1) !'

            assert val.shape[1] == self.n_u, (
                'Expected number of inputs is: {}, but found {}'.format(self.n_u, val.shape[0]))

            # storage
            self._inputs = val

        # error
        else:
            raise ValueError("Inputs cannot be set for system with order <= 1.")


    def set_initial_guess(self):

        initial_cond = self._generate_initial_guess()

        # passing initial cond
        self.simulator.x0 = initial_cond
        self.simulator.set_initial_guess()

        # init time
        self.t0 = 0.0
        history = {}
        history['x0'] = self.states[0, :].reshape((1, -1))
        history['time'] = np.array([[self.t0]])
        history['u0'] = np.full((1, self.n_u), np.nan)
        self.history = history

        # flag update
        self.flags.update({
            'initial_condition_ready': True,
        })

        # end
        return None

    def _generate_initial_guess(self):

        assert self.flags['simulator_ready'], \
            "do_mpc.simulator not generated! Use create_simulator() to generate a simulator."

        states = self.states


        # if order is 2 or more, only then previous inputs are needed
        if self.order > 1:
            inputs = self.inputs
            init_states = states.reshape((-1, 1))
            init_inputs = inputs.reshape((-1, 1))
            initial_cond = np.vstack([init_states, init_inputs])

        else:
            initial_cond = states.reshape((-1, 1))

        #storage
        self.initial_cond = initial_cond

        # end
        return initial_cond


    def make_step(self, u0):
        assert self.flags['initial_condition_ready'] == True, \
            'Surrogate model initial condition not set! Set initial condition!'
        assert u0.shape == (1, self.n_u), \
            f'Input shape is incorrect! Expected shape is (1, {self.n_u}).'

        # init
        #initial_cond = self.initial_cond
        #n_x = self.n_x
        #order = self.order

        ## segregating states and inputs
        #states = initial_cond[0:n_x * order, :]
        #inputs = initial_cond[n_x * order:, :]

        x_full = self.simulator.make_step(u0=u0.reshape((-1, 1)))

        x0 = x_full[0:self.n_x,]

        ## pushing oldest state out of system and inserting the current state
        #new_states = np.vstack([x0, states[0:(self.order - 1) * self.n_x, :]])

        ## setting new initial conditions
        #if self.order > 1:

        #    # pushing oldest input out of system and inserting the current input
        #    new_inputs = np.vstack([u0, inputs[0:(self.order - 2) * self.n_u, :]])

        #    # setting new initial guess by removing the last timestamp data
        #    self.states = self.reshape(new_states, shape=(self.n_x, -1))
        #    self.inputs = self.reshape(new_inputs, shape=(self.n_u, -1))
        #    self._generate_initial_guess()

        #else:
        #    self.states = self.reshape(new_states, shape=(self.n_x, -1))
        #    self._generate_initial_guess()

        # storing simulation history

        # step up time
        self.t0 = self.t0 + self.t_step

        history = self.history
        history['x0'] = np.vstack([history['x0'], x0.reshape((1, -1))])
        history['time'] = np.vstack([history['time'], self.t0])
        history['u0'] = np.vstack([history['u0'], u0.reshape((1, -1))])
        self.history = history

        return x0


    def export_log(self, file_name = 'Surrogate Model Log.csv'):

        assert self.flags['debug'], "Data not logged! Enable debug=True to access log."

        # generating names for the dataframe
        state_names = []
        input_names = []
        for o in range(self.order):
            for n_xn in range(self.n_x):
                state_name = f'state_{n_xn+1}_lag_{o+1}'
                state_names.append(state_name)

            for n_un in range(self.n_u):
                input_name = f'input_{n_un+1}_lag_{o}'
                input_names.append(input_name)

        # dataset
        col_names = ['time'] + state_names + input_names
        data = np.hstack([self.history['time'],
                          self.simulator.data['_x'][:, 0:self.order*self.n_x],
                          self.history['u0'],
                          self.simulator.data['_x'][:, self.order*self.n_x:]])

        # Converting to dataframe
        df = pd.DataFrame(data, columns=col_names)

        # saving dataframe
        df.to_csv(file_name, index=False)

        # end
        return None

