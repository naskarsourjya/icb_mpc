import torch
import do_mpc
import casadi as ca
import  numpy as np
import pandas as pd
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

    def narx_2_dompc_model(self, narx):

        # init
        model = do_mpc.model.Model(model_type='discrete', symvar_type='SX')
        layer_counter = 0

        # variable setup
        system_state = model.set_variable(var_type='_x', var_name='system_state',
                                          shape=(self.order * self.n_x + (self.order - 1) * self.n_u, 1))
        system_input = model.set_variable(var_type='_u', var_name='system_input',
                                          shape=(self.n_u, 1))

        # used by random state tracking algo
        state_ref = model.set_variable(var_type='_tvp', var_name='state_ref', shape=(self.n_x, 1))

        # building input layer of narx
        states_history = system_state[0:self.order * self.n_x]
        inputs_histroy = system_state[self.order * self.n_x:]

        # narx input layer
        input_layer = ca.vertcat(states_history, system_input, inputs_histroy)

        # scaled input layer
        input_layer_scaled = self.scale_input_layer(input_layer, narx.scaler)

        # reading the layers and the biases
        for layer in narx.model.network:

            # linear transformations
            if isinstance(layer, torch.nn.Linear):
                # extracting weight and bias
                weight = layer.weight.cpu().detach().numpy()
                bias = layer.bias.cpu().detach().numpy()

                if layer_counter == 0:
                    output_layer = ca.mtimes(weight, input_layer_scaled) + bias

                else:
                    output_layer = ca.mtimes(weight, output_layer) + bias

                layer_counter += 1

            elif isinstance(layer, torch.nn.Tanh):
                output_layer = ca.tanh(output_layer)

            else:
                raise RuntimeError('{} not supported!'.format(layer))

        # merging the model equations and the history shifting for the rhs
        for i in range((2*self.order) - 1):

            # model euqtions
            if i == 0:
                rhs = output_layer

            # state history shifting
            elif i < self.order:
                start = (i-1)*self.n_x
                end = (i)*self.n_x
                rhs = ca.vertcat(rhs, system_state[start:end])

            # previous input
            elif i == self.order:
                rhs = ca.vertcat(rhs, system_input)

            # input history shifting
            else:
                start = self.order*self.n_x + (i -1 - self.order)*self.n_u
                end = self.order*self.n_x + (i - self.order)*self.n_u
                rhs = ca.vertcat(rhs, system_state[start:end])

        # setting rhs
        model.set_rhs('system_state', rhs)
        model.setup()

        # storage
        self.model = model

        # flag update
        self.flags.update({
            'model_ready': True,
        })

        # end
        return model


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

        assert val.shape[1] == self.order, \
            'Number of samples must be equal to the order of the NARX model!'

        assert val.shape[0] == self.n_x, (
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

            assert self.order - 1 == val.shape[1], \
                'Number of samples for inputs should be (order-1) !'

            assert val.shape[0] == self.n_u, (
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
        inputs = self.inputs

        state_order = self.order
        input_order = self.order - 1

        state_samples = states.shape[1]
        input_samples = inputs.shape[1]

        # ensuring this is the current input
        # stacking states and inputs with order
        order_states = np.vstack([states[:, state_order - i - 1:state_samples - i] for i in range(state_order)])

        # if order is 2 or more, only then previous inputs are needed
        if self.order > 1:
            order_inputs = np.vstack([inputs[:, input_order - i - 1:input_samples - i] for i in range(input_order)])

            # stacking states and inputs for narx model
            initial_cond = np.vstack([order_states, order_inputs])

        else:
            initial_cond = order_states

        #storage
        self.initial_cond = initial_cond

        # end
        return initial_cond


    def make_step(self, u0):
        assert self.flags['initial_condition_ready'] == True, \
            'Surrogate model initial condition not set! Set initial condition!'

        # init
        #initial_cond = self.initial_cond
        #n_x = self.n_x
        #order = self.order

        ## segregating states and inputs
        #states = initial_cond[0:n_x * order, :]
        #inputs = initial_cond[n_x * order:, :]

        x_full = self.simulator.make_step(u0=u0)

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
        if self.history == None:
            history = {}
            history['x0'] = x0
            history['time'] = [0.0]
            history['u0'] = u0

            self.history = history

        else:
            history = self.history

            history['x0'] = np.hstack([history['x0'], x0])
            history['time'].append(history['time'][-1] + self.t_step)
            history['u0'] = np.hstack([history['u0'], u0])

            self.history = history


        if self.flags['debug']:
            # storing simulation history
            if self.log == None:
                log = {}
                log['x_full'] = x_full
                log['u0'] = u0
                log['time'] = [0.0]
                self.log = log

            else:
                log = self.log
                log['x_full'] = np.hstack([log['x_full'], x_full])
                log['u0'] = np.hstack([log['u0'], u0])
                log['time'].append(log['time'][-1] + self.t_step)
                self.log = log

        return x0


    def export_log(self, file_name = 'Surrogate Model Log.csv'):

        assert self.flags['debug'], "Data not logged! Enable debug=True to access log."

        # generating names for the dataframe
        state_names = []
        input_names = []
        for o in range(self.order):
            for n_xn in range(self.n_x):
                state_name = f'state_{n_xn+1}_lag_{self.order-1-o}'
                state_names.append(state_name)

            for n_un in range(self.n_u):
                input_name = f'input_{n_un+1}_lag_{self.order-1-o}'
                input_names.append(input_name)

        # dataset
        col_names = ['time'] + state_names + input_names
        data = np.hstack([np.vstack(self.log['time']),
                          self.log['x_full'].T[:, :self.order*self.n_x],
                          self.log['u0'].T,
                          self.log['x_full'].T[:, self.order*self.n_x:]])

        # Converting to dataframe
        df = pd.DataFrame(data, columns=col_names)

        # saving dataframe
        df.to_csv(file_name, index=False)

        # end
        return None







