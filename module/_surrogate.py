import warnings
import casadi as ca
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from ._springsystem import *
from ._regressor import *

class SurrogateCreator(torch.nn.Module):
    def __init__(self):
        super(SurrogateCreator, self).__init__()

        # data
        self.data = {}
        self.narx_data = {}
        self.narx = {}
        self.surrogate = {}
        self.train_data = {}
        self.test_data = {}
        self.sys = {}

        # generating flags
        self.flags = {
            'data_stored': False,
            'data_split': False,
            'data_preprocessed': False,
            'narx_ready': False,
            'surrogate_ready': False,
            'surrogate_initial_condition_ready': False
        }

        return None


    def random_state_sampler(self, system, n_samples):
        assert self.flags['data_stored'] == False, \
            'Data already exists! Create a new object to create new trajectory.'

        # setting up sysetm
        model= system._get_model()
        simulator = system._get_simulator(model=  model)
        mpc = system._get_random_traj_mpc(model= model)
        estimator = do_mpc.estimator.StateFeedback(model= model)

        # random initial state
        x0 = np.random.uniform(system.lbx, system.ubx).reshape((model.n_x, 1))

        simulator.x0 = x0
        simulator.set_initial_guess()

        mpc.x0 = x0
        mpc.set_initial_guess()

        data_x = [x0]
        data_u = [np.full((model.n_u, 1), np.nan)]
        data_t = [np.array([[0]])]

        for i in tqdm(range(n_samples), desc= 'Generating data'):
            u0 = mpc.make_step(x0)
            y_next = simulator.make_step(u0)
            x0 = estimator.make_step(y_next)

            # check if solver was successful
            if mpc.data['success'].reshape((-1,))[-1] == 0:
                raise RuntimeError('do_mpc did not find a solution for the given problem!')

            # data storage
            data_x.append(x0)
            data_u.append(u0)
            data_t.append(data_t[-1]+ system.t_step)


        data_x = np.concatenate(data_x, axis=1)
        data_u = np.concatenate(data_u, axis=1)
        data_t = np.concatenate(data_t, axis=1)
        self.data['states'] = data_x
        self.data['inputs'] = data_u
        self.data['time'] = data_t
        self.data['n_samples'] = n_samples
        self.data['n_x'] = model.n_x
        self.data['n_u'] = model.n_u
        self.data['n_y'] = model.n_y
        self.data['t_step'] = system.t_step

        self.flags.update({
            'data_stored': True,
        })

        return None

    def random_input_sampler(self, system, n_samples, change_probability = 0.7):
        assert self.flags['data_stored'] == False, \
            'Data already exists! Create a new object to create new trajectory.'

        # setting up sysetm
        model = system._get_model()
        simulator = system._get_simulator(model=model)
        estimator = do_mpc.estimator.StateFeedback(model= model)

        # random initial state
        x0 = np.random.uniform(system.lbx, system.ubx).reshape((model.n_x, 1))

        simulator.x0 = x0
        simulator.set_initial_guess()

        data_x = [x0]
        data_u = [np.full((model.n_u, 1), np.nan)]
        data_t = [np.array([[0]])]

        for i in tqdm(range(n_samples), desc= 'Generating data'):

            # executes if the system decides for a change
            if np.random.rand() < change_probability:
                u0 = np.random.uniform(system.lbu, system.ubu).reshape((-1,1))
                u_prev = u0

            # executes if the system decides to not change
            else:
                u0 = u_prev

            y_next = simulator.make_step(u0)
            x0 = estimator.make_step(y_next)

            # data storage
            data_x.append(x0)
            data_u.append(u0)
            data_t.append(data_t[-1]+system.t_step)


        data_x = np.concatenate(data_x, axis=1)
        data_u = np.concatenate(data_u, axis=1)
        data_t = np.concatenate(data_t, axis=1)

        # storage
        self.data['states'] = data_x
        self.data['inputs'] = data_u
        self.data['time'] = data_t
        self.data['n_samples'] = n_samples
        self.data['n_x'] = model.n_x
        self.data['n_u'] = model.n_u
        self.data['n_y'] = model.n_y
        self.data['t_step'] = system.t_step

        self.flags.update({
            'data_stored': True,
        })

        return None


    def visualize2d(self, system):

        assert self.flags['data_stored'] == True,\
            'Data not found! First run random_trajectory_sampler(), to generate data.'

        model =system._get_model()
        assert model.n_x == 2, 'Only exclusively for systems with 2 states'

        # setting up plot
        fig, ax = plt.subplots(1 + model.n_u)
        fig.suptitle('Input and State space plot')

        ax[0].plot(self.data['states'][0,:], self.data['states'][1,:])

        # Define the limits
        x_lower, x_upper = system.lbx[0], system.ubx[0]  # Limits for the x-axis
        y_lower, y_upper = system.lbx[1], system.ubx[1]  # Limits for the y-axis

        # Plot the box with gray infill
        rect = plt.Rectangle((x_lower, y_lower), x_upper - x_lower, y_upper - y_lower,
                             color='gray', alpha=0.5)
        ax[0].add_patch(rect)

        ax[0].set_xlabel('state 1')
        ax[0].set_ylabel('state 2')

        for i in range(model.n_u):
            ax[i+1].plot(self.data['time'].reshape((-1,)), self.data['inputs'][i, :])
            #ax[i+1].set_xticklabels([])
            upper_limit = np.full((self.data['inputs'][i, :].shape[0],), system.ubu[i])
            lower_limit = np.full((self.data['inputs'][i, :].shape[0],), system.lbu[i])
            # Plot upper and lower limits
            ax[i+1].plot(self.data['time'].reshape((-1,)), upper_limit, linestyle='dashed', color='green')
            ax[i+1].plot(self.data['time'].reshape((-1,)), lower_limit, linestyle='dashed', color='red')

            # gray infill
            ax[i+1].fill_between(self.data['time'].reshape((-1,)), lower_limit, upper_limit, color='gray', alpha=0.5)
            label = 'input_' + str(i + 1)
            ax[i+1].set_ylabel(label)

        ax[-1].set_xlabel('time')
        fig.legend()
        plt.show()

        return None


    def visualize(self, system):
        assert self.flags['data_stored'] == True, \
            'Data not found! First run random_trajectory_sampler(), to generate data.'

        model = system._get_model()

        # setting up plot
        fig, ax = plt.subplots(model.n_x + model.n_u)
        fig.suptitle('Input and State space plot')

        for i in range(model.n_x):
            ax[i].plot(self.data['time'].reshape((-1,)), self.data['states'][i, :])
            upper_limit = np.full((self.data['states'][i, :].shape[0],),
                                system.ubx[i])
            lower_limit = np.full((self.data['states'][i, :].shape[0],),
                                system.lbx[i])
            # Plot upper and lower limits
            ax[i].plot(self.data['time'].reshape((-1,)), upper_limit, linestyle='dashed', color='green')
            ax[i].plot(self.data['time'].reshape((-1,)), lower_limit, linestyle='dashed', color='red')

            # gray infill
            ax[i].fill_between(self.data['time'].reshape((-1,)), lower_limit, upper_limit, color='gray', alpha=0.5)
            label = 'state_' + str(i+1)
            ax[i].set_ylabel(label)

        for i in range(model.n_u):
            ax[i+model.n_x].plot(self.data['time'].reshape((-1,)), self.data['inputs'][i, :])

            upper_limit = np.full((self.data['inputs'][i, :].shape[0],), system.ubu[i])
            lower_limit = np.full((self.data['inputs'][i, :].shape[0],), system.lbu[i])
            # Plot upper and lower limits
            ax[i+model.n_x].plot(self.data['time'].reshape((-1,)), upper_limit, linestyle='dashed', color='green')
            ax[i+model.n_x].plot(self.data['time'].reshape((-1,)), lower_limit, linestyle='dashed', color='red')

            # gray infill
            ax[i+model.n_x].fill_between(self.data['time'].reshape((-1,)), lower_limit, upper_limit, color='gray', alpha=0.5)
            label = 'input_' + str(i + 1)
            ax[i + model.n_x].set_ylabel(label)

        ax[-1].set_xlabel('time')
        fig.legend()
        plt.show()

        return None


    def store_raw_data(self, filename):
        assert self.flags['data_stored'] == True, \
            'Data not found! First run random_trajectory_sampler(), to generate data.'

        # dict format
        storer = {'data': self.data}

        # Save dictionary to pickle file
        with open(filename, "wb") as file:  # Open the file in write-binary mode
            pickle.dump(storer, file)

        # end
        return None


    def load_raw_data(self, filename):

        assert self.flags['data_stored'] == False, \
            'Data already exists! Create a new object to load data.'
         # read file
        with open(filename, "rb") as file:  # Open the file in read-binary mode
            storer = pickle.load(file)

        # store data
        self.data = storer['data']
        #self.system = storer['system']

        # update flag
        self.flags.update({
            'data_stored': True,
        })

        return None

    def data_test_train_splitter(self, train_split= 0.2):
        assert self.flags['data_stored'] == True, \
            'Data not found! First run random_trajectory_sampler(), to generate data.'

        #self.data['states'] = data_x
        #self.data['inputs'] = data_u
        #self.data['time'] = data_t
        #self.data['n_samples'] = n_samples
        #self.data['n_x'] = self.system.model.n_x
        #self.data['n_u'] = self.system.model.n_u
        #self.data['n_y'] = self.system.model.n_y
        #self.data['t_step'] = self.system.t_step

        train_split_len = int(train_split*self.data['n_samples'])

        # splitting series
        self.train_data['states'] = self.data['states'][:, :train_split_len]
        self.train_data['inputs'] = self.data['inputs'][:, :train_split_len]

        self.test_data['states'] = self.data['states'][:, train_split_len:]
        self.test_data['inputs'] = self.data['inputs'][:, train_split_len:]

        # flag update
        self.flags.update({
            'data_split': True,
        })

        return None


    def input_preprocessing(self, states, inputs):
        assert states.shape[0] == self.narx['n_x'], (
            'Expected number of states is: {}, but found {}'.format(self.data['n_x'], states.shape[0]))

        assert inputs.shape[0] == self.narx['n_u'], (
            'Expected number of inputs is: {}, but found {}'.format(self.data['n_u'], inputs.shape[0]))


        order = self.narx['order']
        n_samples = states.shape[1] - 1

        # stacking states and inputs with order
        order_states = np.vstack([states[:,order - 1 - i:n_samples-i] for i in range(order)])
        order_inputs = np.vstack([inputs[:,order - i:n_samples-i +1] for i in range(order)])

        # stacking states and inputs for narx model
        narx_input = np.vstack([order_states, order_inputs])

        # end
        return narx_input


    def output_preprocessing(self, states):
        assert states.shape[0] == self.narx['n_x'], (
            'Expected number of states is: {}, but found {}'.format(self.data['n_x'], states.shape[0]))

        # data gen
        narx_output = states[:,self.narx['order']:]

        # end
        return narx_output

    def data_preprocessing(self, order):
        assert order >= 1, 'Please ensure order is an integer greater than or equal to 1!'
        assert self.flags['data_split'] == True, 'Splitted data not found!'

        # store order
        self.narx['order'] = order
        self.narx['n_samples'] = self.train_data['states'].shape[1] - 1
        self.narx['n_x'] = self.data['n_x']
        self.narx['n_u'] = self.data['n_u']
        self.narx['n_y'] = self.data['n_y']
        self.narx['t_step'] = self.data['t_step']


        # preparing data for NARX
        narx_input = self.input_preprocessing(states=self.train_data['states'], inputs=self.train_data['inputs'])
        narx_output = self.output_preprocessing(states=self.train_data['states'])

        # storage
        self.narx_data['inputs'] = narx_input
        self.narx_data['input_shape'] = narx_input.shape[0]
        self.narx_data['output'] = narx_output
        self.narx_data['output_shape'] = narx_output.shape[0]
        self.narx_data['t_step'] = self.data['t_step']

        # flag update
        self.flags.update({
           'data_preprocessed': True,
        })

        return None


    def _set_device(self, torch_device):

        torch.set_default_device(torch_device)

        return None

    def narx_trainer(self, order, hidden_layers, batch_size, learning_rate, epochs, val=0.1, device = 'auto'):
        assert self.flags['data_stored'] == True, \
            'Data does not exist! Generate or load data!'



        self.data_preprocessing(order=order)

        narx_model = Regressor(input_size= self.narx['order']*(self.narx['n_x'] + self.narx['n_u']),
                                output_size= self.narx['n_x'],
                                hidden_layers=hidden_layers, device=device)

        self._set_device(torch_device= narx_model.torch_device)

        X = torch.tensor(self.narx_data['inputs'].T, dtype=torch.float32)
        Y = torch.tensor(self.narx_data['output'].T, dtype=torch.float32)

        # Create data loader
        dataset = torch.utils.data.TensorDataset(X, Y)
        training_data, test_data = torch.utils.data.random_split(dataset, [1 - val, val],
                                                            generator=torch.Generator(device=narx_model.torch_device))
        train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True,
                                                  generator= torch.Generator(device=narx_model.torch_device))
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,
                                                       generator=torch.Generator(device=narx_model.torch_device))

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(narx_model.parameters(), lr=learning_rate)
        val_loss = -1
        for epoch in tqdm(range(epochs), desc= 'Training NARX:'):
            for batch_X, batch_Y in train_dataloader:
                # Forward pass
                predictions = narx_model(batch_X).squeeze()
                loss = criterion(predictions, batch_Y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            val_loss = 0
            for batch_X, batch_Y in test_dataloader:
                with torch.no_grad():
                    predictions = narx_model(batch_X).squeeze()
                    val_loss += criterion(predictions, batch_Y).item()
            #print(val_loss)

        # store model
        self.narx['model'] = narx_model
        self.narx['hidden_layers'] = hidden_layers

        # flag update
        self.flags.update({
            'narx_ready': True,
        })
        return None


    def save_narx(self, filename):
        assert self.flags['narx_ready'], 'NARX model not found!'

        # dict format
        storer = {'narx': self.narx}

        # Save dictionary to pickle file
        with open(filename, "wb") as file:  # Open the file in write-binary mode
            pickle.dump(storer, file)

        # end
        return None

    def load_narx(self, filename):
        assert self.flags['narx_ready'] == False, \
            'NARX model already exists! Create a new object to load narx.'

        # read file
        with open(filename, "rb") as file:  # Open the file in read-binary mode
            storer = pickle.load(file)

        # store data
        self.narx = storer['narx']
        # self.system = storer['system']

        # update flag
        self.flags.update({
            'narx_ready': True,
        })

        return None


    @torch.no_grad()
    def narx_make_step(self, states, inputs):
        assert self.flags['narx_ready'] == True, 'NARX model not found. Generate or load NARX model!'

        assert states.shape[1] == inputs.shape[1], 'Number of samples for both states and inputs should match!'

        assert states.shape[1] >= self.narx['order'], \
            'Number of samples must exceed or be equal to the order of the NARX model!'

        assert states.shape[0] == self.narx['n_x'], (
            'Expected number of states is: {}, but found {}'.format(self.data['n_x'], states.shape[0]))

        assert inputs.shape[0] == self.narx['n_u'], (
            'Expected number of inputs is: {}, but found {}'.format(self.data['n_u'], inputs.shape[0]))

        order = self.narx['order']

        self._set_device(torch_device=self.narx['model'].torch_device)

        n_samples = states.shape[1]

        # ensuring this is the current input
        # stacking states and inputs with order
        order_states = np.vstack([states[:,order-i-1:n_samples-i] for i in range(order)])
        order_inputs = np.vstack([inputs[:,order-i-1:n_samples-i] for i in range(order)])
        # stacking states and inputs for narx model
        narx_input = np.vstack([order_states, order_inputs])

        #narx_input = self.input_preprocessing(states=order_states, inputs=order_inputs)
        X = torch.tensor(narx_input.T, dtype=torch.float32)
        #X_set = torch.utils.data.TensorDataset(X)

        Y_pred = self.narx['model'](X).cpu().numpy().T

        return Y_pred

    def narx_2_dompc(self):
        # sanity
        assert self.flags['narx_ready'] == True, 'NARX model not found. Generate or load NARX model!'

        # model setup
        # init
        model = do_mpc.model.Model(model_type='discrete', symvar_type='SX')

        # variable setup
        system_state = model.set_variable(var_type='_x', var_name='system_state',
                                          shape=(self.narx['order'] * self.narx['n_x'] + (self.narx['order'] - 1) * self.narx['n_u'], 1))
        system_input = model.set_variable(var_type='_u', var_name='system_input',
                                          shape=(self.narx['n_u'], 1))

        states_history = system_state[0:self.narx['order'] * self.narx['n_x']]
        inputs_histroy = system_state[self.narx['order'] * self.narx['n_x']:]
        input_layer = ca.vertcat(states_history, system_input, inputs_histroy)
        layer_counter = 0

        # reading the layers and the biases
        for layer in self.narx['model'].network:

            # linear transformations
            if isinstance(layer, torch.nn.Linear):
                # extracting weight and bias
                weight = layer.weight.cpu().detach().numpy()
                bias = layer.bias.cpu().detach().numpy()

                if layer_counter == 0:
                    output_layer = ca.mtimes(weight, input_layer) + bias

                else:
                    output_layer = ca.mtimes(weight, output_layer) + bias

                layer_counter += 1

            elif isinstance(layer, torch.nn.Tanh):
                output_layer = ca.tanh(output_layer)

            else:
                raise RuntimeError('{} not supported!'.format(layer))

        # merging the model equations and the history shifting for the rhs
        for i in range((2*self.narx['order']) - 1):

            # model euqtions
            if i == 0:
                rhs = output_layer

            # input shifting
            elif i == self.narx['order']:
                rhs = ca.vertcat(rhs, system_input)

            # state history shifting
            elif i < self.narx['order']:
                start = (i-1)*self.narx['n_x']
                end = (i)*self.narx['n_x']
                rhs = ca.vertcat(rhs, system_state[start:end])

            # input history shifting
            else:
                start = self.narx['order']*self.narx['n_x'] + (i -1 - self.narx['order'])*self.narx['n_u']
                end = self.narx['order']*self.narx['n_x'] + (i - self.narx['order'])*self.narx['n_u']
                rhs = ca.vertcat(rhs, system_state[start:end])

        # setting rhs
        model.set_rhs('system_state', rhs)
        model.setup()


        # simulator setup
        # init
        simulator = do_mpc.simulator.Simulator(model)
        simulator.set_param(t_step = self.narx['t_step'])
        simulator.setup()

        # storage
        self.surrogate['model'] = model
        self.surrogate['simulator'] = simulator

        self.surrogate['order'] = self.narx['order']
        self.surrogate['n_x'] = self.narx['n_x']
        self.surrogate['n_u'] = self.narx['n_u']
        self.surrogate['n_y'] = self.narx['n_y']
        self.surrogate['t_step'] = self.narx['t_step']

        # flag update
        self.flags.update({
            'surrogate_ready': True,
        })

        # end
        return None

    def save_surrogate(self, filename):
        assert self.flags['surrogate_ready'] == True, 'Surrogate model not found. Generate Surrogate model!'

        # dict format
        storer = {'surrogate': self.surrogate}

        # Save dictionary to pickle file
        with open(filename, "wb") as file:  # Open the file in write-binary mode
            pickle.dump(storer, file)

        # end
        return None

    def load_surrogate(self, filename):
        assert self.flags['surrogate_ready'] == False, \
            'Surrogate model already exists! Create a new object to load surrogate.'

        # read file
        with open(filename, "rb") as file:  # Open the file in read-binary mode
            storer = pickle.load(file)

        # store data
        self.surrogate = storer['narx']
        # self.system = storer['system']

        # update flag
        self.flags.update({
            'surrogate_ready': True,
        })

        return None

    def simulator_set_initial_guess(self, states, inputs):
        assert self.flags['surrogate_ready'] == True, 'Surrogate model not found. Generate or load Surrogate model!'

        assert states.shape[1] - 1 == inputs.shape[1], \
            'Number of samples for states should exceed that of inputs by one!'

        assert states.shape[1] == self.surrogate['order'], \
            'Number of samples must be equal to the order of the NARX model!'

        assert states.shape[0] == self.surrogate['n_x'], (
            'Expected number of states is: {}, but found {}'.format(self.data['n_x'], states.shape[0]))

        assert inputs.shape[0] == self.surrogate['n_u'], (
            'Expected number of inputs is: {}, but found {}'.format(self.data['n_u'], inputs.shape[0]))

        state_order = self.surrogate['order']
        input_order = self.surrogate['order'] - 1

        state_samples = states.shape[1]
        input_samples = inputs.shape[1]

        # ensuring this is the current input
        # stacking states and inputs with order
        order_states = np.vstack([states[:, state_order - i - 1:state_samples - i] for i in range(state_order)])
        order_inputs = np.vstack([inputs[:, input_order - i - 1:input_samples - i] for i in range(input_order)])

        # stacking states and inputs for narx model
        initial_cond = np.vstack([order_states, order_inputs])

        # passing initial cond
        self.surrogate['simulator'].x0 = initial_cond
        self.surrogate['simulator'].set_initial_guess()

        # flag update
        self.flags.update({
            'surrogate_initial_condition_ready': True,
        })


        # end
        return None


    def simulator_make_step(self, u0):
        assert self.flags['surrogate_ready'] == True, 'Surrogate model not found. Generate or load Surrogate model!'

        assert self.flags['surrogate_initial_condition_ready'] == True, \
            'Surrogate model initial condition not set! Generate or load Surrogate model!'

        x_full = self.surrogate['simulator'].make_step(u0= u0)

        x0 = x_full[0:self.surrogate['n_x'],]

        return x0


    def surrogate_error(self, system):
        assert self.flags['data_split'] == True, 'Split data not found!'

        #self.test_data['states'] = self.data['states'][:, train_split_len:]
        #self.test_data['inputs']

        # init
        error = []
        narx_input = self.input_preprocessing(states=self.test_data['states'], inputs=self.test_data['inputs'])
        narx_output = self.output_preprocessing(states=self.test_data['states'])
        n_samples = narx_input.shape[1]


        for i in tqdm(range(n_samples), desc= 'Calculating surrogate model error'):

            # extracting individual elements
            states = narx_input[0:self.narx['order']*self.narx['n_x'],i].reshape((-1, self.narx['order']))
            inputs = narx_input[self.narx['order']*self.narx['n_x']:,i].reshape((-1, self.narx['order']))
            output = narx_output[:, i].reshape((-1,1))

            # extracting inputs
            input_history = inputs[:, 0:-1]
            input = inputs[:, -1].reshape((-1,1))

            # initiating system
            self.narx_2_dompc()
            self.simulator_set_initial_guess(states=states, inputs=input_history)

            # simulating system
            x0 = self.simulator_make_step(u0= input)

            # appending error
            delta = output - x0
            error.append(delta)

        # error np array
        error = np.concatenate(error, axis=1)

        return error


    def train_cqr(self):


        return None