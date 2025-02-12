import warnings
import casadi as ca
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import do_mpc
from tqdm import tqdm
import module
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class SurrogateCreator():
    def __init__(self, set_seed = None):
        # for repetable results
        if set_seed is not None:
            np.random.seed(set_seed)

        # data
        self.set_seed = set_seed
        self.data = {}
        self.narx = {}
        self.surrogate = {}
        self.cqr = {}
        self.mpc={}
        self.simulation={}

        # plottting row size
        self.height_px = 700
        self.width_px = 1800

        # generating flags
        self.flags = {
            'data_stored': False,
            'data_split': False,
            'data_preprocessed': False,
            'narx_ready': False,
            'surrogate_ready': False,
            'surrogate_initial_condition_ready': False,
            'qr_ready': False,
            'cqr_ready': False,
            'cqr_initial_condition_ready': False,
            'mpc_ready': False,
            'mpc_initial_condition_ready': False,
            'simulation_ready': False
        }

        # end of init

    def reshape(self, array, shape):

        # rows and columns
        rows, cols = shape

        # end
        return array.reshape(cols, rows).T

    def random_state_sampler(self, system, n_samples):
        assert self.flags['data_stored'] == False, \
            'Data already exists! Create a new object to create new trajectory.'

        # setting up sysetm
        model= system._get_model()
        simulator = system._get_simulator(model=  model)
        mpc = system._get_random_traj_mpc(model= model)
        estimator = do_mpc.estimator.StateFeedback(model= model)

        # random initial state
        #x0 = np.random.uniform(system.lbx, system.ubx).reshape((model.n_x, 1))
        x0 = self.reshape(np.random.uniform(system.lbx, system.ubx), shape=(model.n_x, 1))

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
        #x0 = np.random.uniform(system.lbx, system.ubx).reshape((model.n_x, 1))
        x0 = self.reshape(np.random.uniform(system.lbx, system.ubx), shape= (model.n_x, 1))

        simulator.x0 = x0
        simulator.set_initial_guess()

        data_x = [x0]
        data_u = [np.full((model.n_u, 1), np.nan)]
        data_t = [np.array([[0]])]

        for i in tqdm(range(n_samples), desc= 'Generating data'):

            # executes if the system decides for a change
            if np.random.rand() < change_probability:
                #u0 = np.random.uniform(system.lbu, system.ubu).reshape((-1,1))
                u0 = self.reshape(np.random.uniform(system.lbu, system.ubu), shape=(-1,1))
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
        self.data['lbx'] = system.lbx
        self.data['ubx'] = system.ubx
        self.data['lbu'] = system.lbu
        self.data['ubu'] = system.ubu

        self.flags.update({
            'data_stored': True,
        })

        return None


    def visualize2d_data(self):

        assert self.flags['data_stored'] == True,\
            'Data not found! First run random_trajectory_sampler(), to generate data.'

        # setting up plot
        fig, ax = plt.subplots(1 + self.data['n_u'], figsize=(24, 6 * (1 + self.data['n_u'])))
        fig.suptitle('Input and State space plot')

        ax[0].plot(self.data['states'][0,:], self.data['states'][1,:])

        # Define the limits
        x_lower, x_upper = self.data['lbx'][0], self.data['ubx'][0]  # Limits for the x-axis
        y_lower, y_upper = self.data['lbx'][1], self.data['ubx'][1]  # Limits for the y-axis

        # Plot the box with gray infill
        rect = plt.Rectangle((x_lower, y_lower), x_upper - x_lower, y_upper - y_lower,
                             color='gray', alpha=0.5)
        ax[0].add_patch(rect)

        ax[0].set_xlabel('state 1')
        ax[0].set_ylabel('state 2')

        for i in range(self.data['n_u']):
            ax[i+1].plot(self.data['time'].reshape((-1,)), self.data['inputs'][i, :])
            #ax[i+1].set_xticklabels([])
            upper_limit = np.full((self.data['inputs'][i, :].shape[0],), self.data['ubu'][i])
            lower_limit = np.full((self.data['inputs'][i, :].shape[0],), self.data['lbu'][i])
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


    def visualize_data(self):
        assert self.flags['data_stored'] == True, \
            'Data not found! First run random_trajectory_sampler(), to generate data.'

        # setting up plot
        fig, ax = plt.subplots(self.data['n_x'] + self.data['n_u'],
                               figsize=(24, 6 * (self.data['n_x'] + self.data['n_u'])))
        fig.suptitle('Input and State space plot')

        for i in range(self.data['n_x']):
            ax[i].plot(self.data['time'].reshape((-1,)), self.data['states'][i, :])
            upper_limit = np.full((self.data['states'][i, :].shape[0],),
                                self.data['ubx'][i])
            lower_limit = np.full((self.data['states'][i, :].shape[0],),
                                self.data['lbx'][i])
            # Plot upper and lower limits
            ax[i].plot(self.data['time'].reshape((-1,)), upper_limit, linestyle='dashed', color='green')
            ax[i].plot(self.data['time'].reshape((-1,)), lower_limit, linestyle='dashed', color='red')

            # gray infill
            ax[i].fill_between(self.data['time'].reshape((-1,)), lower_limit, upper_limit, color='gray', alpha=0.5)
            label = 'state_' + str(i+1)
            ax[i].set_ylabel(label)

        for i in range(self.data['n_u']):
            ax[i+self.data['n_x']].plot(self.data['time'].reshape((-1,)), self.data['inputs'][i, :])

            upper_limit = np.full((self.data['inputs'][i, :].shape[0],), self.data['ubu'][i])
            lower_limit = np.full((self.data['inputs'][i, :].shape[0],), self.data['lbu'][i])
            # Plot upper and lower limits
            ax[i+self.data['n_x']].plot(self.data['time'].reshape((-1,)), upper_limit, linestyle='dashed', color='green')
            ax[i+self.data['n_x']].plot(self.data['time'].reshape((-1,)), lower_limit, linestyle='dashed', color='red')

            # gray infill
            ax[i+self.data['n_x']].fill_between(self.data['time'].reshape((-1,)), lower_limit, upper_limit, color='gray', alpha=0.5)
            label = 'input_' + str(i + 1)
            ax[i + self.data['n_x']].set_ylabel(label)

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


    def input_preprocessing(self, states, inputs):
        assert states.shape[0] == self.data['n_x'], (
            'Expected number of states is: {}, but found {}'.format(self.data['n_x'], states.shape[0]))

        assert inputs.shape[0] == self.data['n_u'], (
            'Expected number of inputs is: {}, but found {}'.format(self.data['n_u'], inputs.shape[0]))


        order = self.data['order']
        n_samples = states.shape[1] - 1

        # stacking states and inputs with order
        order_states = np.vstack([states[:,order - 1 - i:n_samples-i] for i in range(order)])
        order_inputs = np.vstack([inputs[:,order - i:n_samples-i +1] for i in range(order)])

        # stacking states and inputs for narx model
        narx_input = np.vstack([order_states, order_inputs])

        # end
        return narx_input


    def output_preprocessing(self, states):
        assert states.shape[0] == self.data['n_x'], (
            'Expected number of states is: {}, but found {}'.format(self.data['n_x'], states.shape[0]))

        # data gen
        narx_output = states[:,self.data['order']:]

        # end
        return narx_output


    def data_splitter(self, order, narx_train= 0.4,
                      cqr_train= 0.4, cqr_calibration= 0.1, test = 0.1):
        assert self.flags['data_stored'] == True, \
            'Data not found! First run random_trajectory_sampler(), to generate data.'

        assert order >= 1, 'Please ensure order is an integer greater than or equal to 1!'

        assert isinstance(order, int), "Order must be an integer more than or equal to 1!"

        # store order
        self.data['order'] = order

        # concatinating the deta with
        X = self.input_preprocessing(states=self.data['states'], inputs=self.data['inputs']).T
        Y = self.output_preprocessing(states=self.data['states']).T
        t = self.data['time'][:, order:].T

        sets = ['narx_train', 'cqr_train', 'cqr_calibration', 'test']
        ratios = [narx_train, cqr_train, cqr_calibration, test]

        for i, set in enumerate(sets[:-1]):
            input_name = set + '_inputs'
            output_name = set + '_outputs'
            time_name = set + '_timestamps'

            ratio = ratios[i] / sum(ratios[i:])
            if i == len(sets) -2:
                last_input_name = sets[-1] + '_inputs'
                last_output_name = sets[-1] + '_outputs'
                last_time_name = sets[-1] + '_timestamps'

                (self.data[last_input_name], self.data[input_name], self.data[last_output_name], self.data[output_name],
                 self.data[last_time_name], self.data[time_name]) = (
                    train_test_split(X, Y, t, test_size=ratio, random_state=self.set_seed))


            else:
                X, self.data[input_name], Y, self.data[output_name], t, self.data[time_name] = (
                        train_test_split(X, Y, t, test_size=ratio, random_state=self.set_seed))

        for set in sets:
            input_name = set + '_inputs'
            output_name = set + '_outputs'
            time_name = set + '_timestamps'

            self.data[input_name] = self.data[input_name].T
            self.data[output_name] = self.data[output_name].T
            self.data[time_name] = self.data[time_name].T



        # storage
        self.narx['n_samples'] = self.data['narx_train_inputs'].shape[1]
        self.narx['order'] = order
        self.narx['n_x'] = self.data['n_x']
        self.narx['n_u'] = self.data['n_u']
        self.narx['n_y'] = self.data['n_y']
        self.narx['t_step'] = self.data['t_step']

        # flag update
        self.flags.update({
            'data_split': True,
        })

        return None


    def _set_device(self, torch_device):

        torch.set_default_device(torch_device)

        return None


    def narx_trainer(self, hidden_layers, batch_size, learning_rate, epochs= 1000,
                     validation_split = 0.2, scheduler_flag = True, device = 'auto', train_threshold = 1e-8):
        assert self.flags['data_stored'] == True, \
            'Data does not exist! Generate or load data!'

        # init
        narx_model = module.Regressor(input_size= self.narx['order']*(self.narx['n_x'] + self.narx['n_u']),
                                output_size= self.narx['n_x'],
                                hidden_layers=hidden_layers, device=device)
        train_history = {'training_loss': [],
                         'validation_loss': [],
                         'learning_rate': [],
                         'epochs': []}

        # setting computation device
        self._set_device(torch_device= narx_model.torch_device)

        # converting datasets to tensors
        X_torch = torch.tensor(self.data['narx_train_inputs'].T, dtype=torch.float32)
        Y_torch = torch.tensor(self.data['narx_train_outputs'].T, dtype=torch.float32)

        # Create TensorDataset
        dataset = torch.utils.data.TensorDataset(X_torch, Y_torch)

        # splitting full datset
        train_dataset, validation_dataset = (
            torch.utils.data.random_split(dataset= dataset, lengths=[1-validation_split, validation_split],
                            generator=torch.Generator(device=narx_model.torch_device).manual_seed(self.set_seed)))

        # creating DataLoader with batch_size
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            generator= torch.Generator(device=narx_model.torch_device).manual_seed(self.set_seed))
        validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True,
                            generator=torch.Generator(device=narx_model.torch_device).manual_seed(self.set_seed))

        # setting up Mean Squared Error as loss function for training
        criterion = torch.nn.MSELoss()

        # setting up optimiser for training
        optimizer = torch.optim.Adam(narx_model.parameters(), lr=learning_rate)

        # scheduler setup
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        # main training loop
        for epoch in tqdm(range(epochs), desc= 'Training NARX'):

            # narx training
            train_loss = 0
            for batch_X, batch_Y in train_dataloader:

                # Forward pass
                predictions = narx_model(batch_X).squeeze()
                loss = criterion(predictions, batch_Y)

                # Backward pass / parameters update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # storing loss
                train_loss += loss.item()

            # narx validation
            val_loss = 0
            for batch_X, batch_Y in validation_dataloader:
                with torch.no_grad():
                    predictions = narx_model(batch_X).squeeze()
                    val_loss += criterion(predictions, batch_Y).item()

            # storing data
            train_history['training_loss'].append(train_loss)
            train_history['validation_loss'].append(val_loss)
            train_history['epochs'].append(epoch)
            train_history['learning_rate'].append(optimizer.param_groups[0]["lr"])

            # learning rate update
            if scheduler_flag:
                lr_scheduler.step(val_loss)

                # break if training min learning rate is reached
                if optimizer.param_groups[0]["lr"] <= train_threshold:
                    break

        # store model
        self.narx['model'] = narx_model
        self.narx['hidden_layers'] = hidden_layers
        self.narx['train_history'] = train_history
        # flag update
        self.flags.update({
            'narx_ready': True,
        })

        # end
        return None


    def plot_narx_training_history(self):
        assert self.flags['narx_ready'] == True, \
            'NARX not found! Generate or load NARX model!'

        # plot init
        fig, ax = plt.subplots(2, figsize=(24, 6 * 2))
        fig.suptitle('NARX Training History')

        # plot 1: Learning rate plot
        ax[0].plot(self.narx['epochs'], self.narx['learning_rate'], label='Learning Rate')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Learning Rate')
        ax[0].set_yscale('log')
        ax[0].grid()


        # plot 2: Training Loss plots
        ax[1].plot(self.narx['epochs'], self.narx['training_loss'], color='blue', label='Training Loss')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Training Loss', color='blue')
        ax[1].set_yscale('log')
        ax[1].tick_params(axis='y', labelcolor='blue')
        ax[1].grid()

        # plot 2: Validation Plot
        ax_n = ax[1].twinx()
        ax_n.plot(self.narx['epochs'], self.narx['validation_loss'], color='red', label='Validation Loss')
        ax_n.tick_params(axis='y', labelcolor='red')
        ax_n.set_ylabel('Validation Loss', color='red')
        ax_n.set_yscale('log')


        # show plot
        plt.show()

        # end
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


    #@torch.no_grad()
    def narx_make_step(self, states, inputs):
        assert self.flags['narx_ready'] == True, 'NARX model not found. Generate or load NARX model!'

        assert states.shape[1] == inputs.shape[1], 'Number of samples for both states and inputs should match!'

        assert states.shape[1] == self.narx['order'], \
            'Number of samples must be equal to the order of the NARX model!'

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
        with torch.no_grad():
            Y_pred = self.narx['model'](X).cpu().numpy().T

        return Y_pred

    def narx_2_dompc(self):
        # sanity
        assert self.flags['narx_ready'] == True, 'NARX model not found. Generate or load NARX model!'

        # model setup
        # init
        model = do_mpc.model.Model(model_type='discrete', symvar_type='SX')
        layer_counter = 0

        # variable setup
        system_state = model.set_variable(var_type='_x', var_name='system_state',
                                          shape=(self.narx['order'] * self.narx['n_x'] + (self.narx['order'] - 1) * self.narx['n_u'], 1))
        system_input = model.set_variable(var_type='_u', var_name='system_input',
                                          shape=(self.narx['n_u'], 1))

        # used by random state tracking algo
        state_ref = model.set_variable(var_type='_tvp', var_name='state_ref', shape=(self.narx['n_x'], 1))

        # building input layer of narx
        states_history = system_state[0:self.narx['order'] * self.narx['n_x']]
        inputs_histroy = system_state[self.narx['order'] * self.narx['n_x']:]

        # narx input layer
        input_layer = ca.vertcat(states_history, system_input, inputs_histroy)

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

            # state history shifting
            elif i < self.narx['order']:
                start = (i-1)*self.narx['n_x']
                end = (i)*self.narx['n_x']
                rhs = ca.vertcat(rhs, system_state[start:end])

            # previous input
            elif i == self.narx['order']:
                rhs = ca.vertcat(rhs, system_input)

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
        tvp_template = simulator.get_tvp_template()
        def tvp_fun(t_ind):
            return tvp_template
        simulator.set_tvp_fun(tvp_fun)
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
        return model, simulator

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

    def simulator_set_initial_guess(self, states, inputs=None):
        assert self.flags['surrogate_ready'] == True, 'Surrogate model not found. Generate or load Surrogate model!'

        assert states.shape[1] == self.surrogate['order'], \
            'Number of samples must be equal to the order of the NARX model!'

        assert states.shape[0] == self.surrogate['n_x'], (
            'Expected number of states is: {}, but found {}'.format(self.data['n_x'], states.shape[0]))

        if self.surrogate['order']>1:
            assert isinstance(inputs, np.ndarray), "If order is more than 1, then input is needed!"

            assert states.shape[1] - 1 == inputs.shape[1], \
                'Number of samples for states should exceed that of inputs by one!'

            assert inputs.shape[0] == self.surrogate['n_u'], (
                'Expected number of inputs is: {}, but found {}'.format(self.data['n_u'], inputs.shape[0]))

        state_order = self.surrogate['order']
        input_order = self.surrogate['order'] - 1

        state_samples = states.shape[1]
        input_samples = inputs.shape[1]

        # ensuring this is the current input
        # stacking states and inputs with order
        order_states = np.vstack([states[:, state_order - i - 1:state_samples - i] for i in range(state_order)])

        # if order is 2 or more, only then previous inputs are needed
        if self.surrogate['order'] > 1:
            order_inputs = np.vstack([inputs[:, input_order - i - 1:input_samples - i] for i in range(input_order)])

            # stacking states and inputs for narx model
            initial_cond = np.vstack([order_states, order_inputs])

        else:
            initial_cond = order_states

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


    def surrogate_error(self, cqr_train_inputs, cqr_train_outputs):
        assert self.flags['data_split'] == True, 'Split data not found!'
        assert self.flags['narx_ready'] == True, 'NARX model not found. Generate or load NARX model!'

        # init
        error = []
        n_samples = cqr_train_inputs.shape[1]

        # calculating error
        for i in tqdm(range(n_samples), desc= 'Calculating surrogate model error'):

            # extracting individual elements
            states = self.reshape(cqr_train_inputs[0:self.data['order']*self.data['n_x'],i],
                                  shape=(self.data['n_x'], self.data['order']))
            inputs = self.reshape(cqr_train_inputs[self.data['order'] * self.data['n_x']:, i],
                                  shape=(self.data['n_u'], self.data['order']))
            output = self.reshape(cqr_train_outputs[:, i],
                                  shape=(self.data['n_x'], 1))


            # extracting inputs
            input_history = inputs[:, 0:-1]
            input = self.reshape(inputs[:, -1], shape=(-1, 1))

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

        # end
        return error

    def _pinball_loss(self, y, y_hat, quantile):

        #if y > y_hat:
        #    loss = quantile * (y - y_hat)
        #else:
        #    loss = (1 - quantile) * (y_hat - y)

        # converting to scalar
        #mean_loss = torch.mean(loss)

        diff = y - y_hat
        loss = torch.maximum(quantile * diff, (quantile - 1) * diff)
        mean_loss = loss.mean()

        # end
        return mean_loss


    def train_individual_qr(self, alpha, hidden_layers, device = 'auto', learning_rate= 0.1, batch_size= 32,
                  validation_split= 0.2, scheduler_flag= True, epochs = 1000, train_threshold = 1e-8):

        assert self.flags['data_split'] == True, 'Split data not found!'
        assert 0 < alpha < 1, "All alpha must be between 0 and 1"

        # calculate the surrogate model error
        self.data['cqr_train_errors'] = self.surrogate_error(cqr_train_inputs= self.data['cqr_train_inputs'],
                                                             cqr_train_outputs= self.data['cqr_train_outputs'])

        # init
        models = []
        history_list = []

        # generating quantiles
        low_quantile = alpha/2
        high_quantile = 1-alpha/2
        quantiles = [high_quantile] + [low_quantile]
        n_q = len(quantiles)

        # creating a model for each quantile
        for quantile in quantiles:

            # model init
            cqr_model_n = module.Regressor(input_size= self.narx['order']*(self.narx['n_x'] + self.narx['n_u']),
                                output_size= self.narx['n_x'],
                                hidden_layers=hidden_layers, device=device)

            # setting training history
            train_history = {'training_loss': [],
                             'validation_loss': [],
                             'learning_rate': [],
                             'epochs': [],
                             'quantile': []}

            # setting computation device
            self._set_device(torch_device=cqr_model_n.torch_device)

            # converting datasets to tensors
            X_torch = torch.tensor(self.data['cqr_train_inputs'].T, dtype=torch.float32)
            Y_torch = torch.tensor(self.data['cqr_train_errors'].T, dtype=torch.float32)

            # Create TensorDataset
            dataset = torch.utils.data.TensorDataset(X_torch, Y_torch)

            # splitting full datset
            train_dataset, validation_dataset = (
                torch.utils.data.random_split(dataset=dataset, lengths=[1 - validation_split, validation_split],
                                              generator=torch.Generator(device=cqr_model_n.torch_device).manual_seed(
                                                  self.set_seed)))

            # creating DataLoader with batch_size
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                           generator=torch.Generator(
                                                               device=cqr_model_n.torch_device).manual_seed(
                                                               self.set_seed))
            validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True,
                                                                generator=torch.Generator(
                                                                    device=cqr_model_n.torch_device).manual_seed(
                                                                    self.set_seed))

            # setting up optimiser for training
            optimizer = torch.optim.Adam(cqr_model_n.parameters(), lr=learning_rate)

            # scheduler setup
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

            # main training loop
            for epoch in tqdm(range(epochs), desc=f'Training Cqr q= {quantile}'):

                # cqr training
                train_loss = 0
                for batch_X, batch_Y in train_dataloader:
                    # Forward pass
                    Y_hat = cqr_model_n(batch_X).squeeze()
                    loss = self._pinball_loss(y=batch_Y, y_hat=Y_hat, quantile=quantile)

                    # Backward pass / parameters update
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # storing loss
                    train_loss += loss.item()

                # narx validation
                val_loss = 0
                for batch_X, batch_Y in validation_dataloader:
                    with torch.no_grad():
                        Y_hat = cqr_model_n(batch_X).squeeze()
                        val_loss += self._pinball_loss(y= batch_Y, y_hat= Y_hat, quantile=quantile).item()

                # storing data
                train_history['quantile'].append(quantile)
                train_history['training_loss'].append(train_loss)
                train_history['validation_loss'].append(val_loss)
                train_history['epochs'].append(epoch)
                train_history['learning_rate'].append(optimizer.param_groups[0]["lr"])

                # learning rate update
                if scheduler_flag:
                    lr_scheduler.step(val_loss)

                    # break if training min learning rate is reached
                    if optimizer.param_groups[0]["lr"] <= train_threshold:
                        break

            # storage
            models.append(cqr_model_n)
            history_list.append(train_history)

        # creating one merged model
        cqr_model = module.MergedModel(models= models, device= device)

        # inserting the mean prediction model
        full_model_list = [self.narx['model']] + models
        full_model = module.MergedModel(models= full_model_list, device= device)

        # store model
        self.cqr['cqr_model'] = cqr_model
        self.cqr['full_model'] = full_model
        self.cqr['train_history_list'] = history_list
        self.cqr['alpha'] = alpha
        self.cqr['quantiles'] = quantiles
        self.cqr['low_quantile'] = low_quantile
        self.cqr['high_quantile'] = high_quantile
        self.cqr['order'] = self.data['order']
        self.cqr['n_x'] = self.data['n_x']
        self.cqr['n_u'] = self.data['n_u']
        self.cqr['n_y'] = self.data['n_y']
        self.cqr['n_q'] = n_q
        self.cqr['t_step'] = self.data['t_step']
        self.cqr['type'] = 'individual'

        # flag update
        self.flags.update({
            'qr_ready': True,
        })

        # end
        return None


    def train_all_qr(self, alpha, hidden_layers, device = 'auto', learning_rate= 0.1, batch_size= 32,
                  validation_split= 0.2, scheduler_flag= True, epochs = 1000, train_threshold = 1e-8):

        assert self.flags['data_split'] == True, 'Split data not found!'
        assert 0 < alpha < 1, "All alpha must be between 0 and 1"

        # calculate the surrogate model error on cqr training data
        self.data['cqr_train_errors'] = self.surrogate_error(cqr_train_inputs= self.data['cqr_train_inputs'],
                                                             cqr_train_outputs= self.data['cqr_train_outputs'])

        # init
        models = []
        history_list = []

        # generating quantiles
        low_quantile = alpha / 2
        high_quantile = 1 - alpha / 2
        quantiles = [high_quantile] + [low_quantile]
        n_q = len(quantiles)

        # model init
        cqr_model = module.Regressor(input_size=self.narx['order'] * (self.narx['n_x'] + self.narx['n_u']),
                                output_size=self.narx['n_x'] * n_q,
                                hidden_layers=hidden_layers, device=device)

        # setting training history
        train_history = {'training_loss': [],
                         'validation_loss': [],
                         'learning_rate': [],
                         'epochs': [],
                         'quantile': []}

        # setting computation device
        self._set_device(torch_device=cqr_model.torch_device)

        # converting datasets to tensors
        X_torch = torch.tensor(self.data['cqr_train_inputs'].T, dtype=torch.float32)

        # stacking once per quantile
        Y_stacked = np.vstack([self.data['cqr_train_errors'] for _ in range(n_q)])

        # converting to tensor
        Y_torch = torch.tensor(Y_stacked.T, dtype=torch.float32)

        # Create TensorDataset
        dataset = torch.utils.data.TensorDataset(X_torch, Y_torch)

        # splitting full datset
        train_dataset, validation_dataset = (
            torch.utils.data.random_split(dataset=dataset, lengths=[1 - validation_split, validation_split],
                                          generator=torch.Generator(device=cqr_model.torch_device).manual_seed(
                                              self.set_seed)))

        # creating DataLoader with batch_size
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                       generator=torch.Generator(
                                                           device=cqr_model.torch_device).manual_seed(
                                                           self.set_seed))
        validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True,
                                                            generator=torch.Generator(
                                                                device=cqr_model.torch_device).manual_seed(
                                                                self.set_seed))

        # setting up optimiser for training
        optimizer = torch.optim.Adam(cqr_model.parameters(), lr=learning_rate)

        # scheduler setup
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        # main training loop
        for epoch in tqdm(range(epochs), desc=f'Training All CQR'):

            # cqr training
            train_loss = 0
            for batch_X, batch_Y in train_dataloader:
                # Forward pass
                Y_hat = cqr_model(batch_X).squeeze()
                loss = 0
                for quantile in quantiles:
                    loss += self._pinball_loss(y=batch_Y, y_hat=Y_hat, quantile=quantile)

                # Backward pass / parameters update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # storing loss
                train_loss += loss.item()

            # narx validation
            val_loss = 0
            for batch_X, batch_Y in validation_dataloader:
                with torch.no_grad():
                    Y_hat = cqr_model(batch_X).squeeze()
                    for quantile in quantiles:
                        val_loss += self._pinball_loss(y=batch_Y, y_hat=Y_hat, quantile=quantile).item()

            # storing data
            train_history['quantile'].append(quantile)
            train_history['training_loss'].append(train_loss)
            train_history['validation_loss'].append(val_loss)
            train_history['epochs'].append(epoch)
            train_history['learning_rate'].append(optimizer.param_groups[0]["lr"])

            # learning rate update
            if scheduler_flag:
                lr_scheduler.step(val_loss)

                # break if training min learning rate is reached
                if optimizer.param_groups[0]["lr"] <= train_threshold:
                    break

        # storage
        history_list.append(train_history)

        # inserting the mean prediction model
        full_model_list = [self.narx['model']] + [cqr_model]
        full_model = module.MergedModel(models=full_model_list, device=device)

        # store model
        self.cqr['cqr_model'] = cqr_model
        self.cqr['full_model'] = full_model
        self.cqr['train_history_list'] = history_list
        self.cqr['alpha'] = alpha
        self.cqr['quantiles'] = quantiles
        self.cqr['low_quantile'] = low_quantile
        self.cqr['high_quantile'] = high_quantile
        self.cqr['order'] = self.data['order']
        self.cqr['n_x'] = self.data['n_x']
        self.cqr['n_u'] = self.data['n_u']
        self.cqr['n_y'] = self.data['n_y']
        self.cqr['n_q'] = n_q
        self.cqr['t_step'] = self.data['t_step']
        self.cqr['type'] = 'all'

        # flag update
        self.flags.update({
            'qr_ready': True,
        })

        # end
        return None


    def save_cqr(self, filename):
        assert self.flags['cqr_ready'] and self.flags['qr_ready'], 'CQR model not found!'

        # dict format
        storer = {'cqr': self.cqr}

        # Save dictionary to pickle file
        with open(filename, "wb") as file:  # Open the file in write-binary mode
            pickle.dump(storer, file)

        # end
        return None

    def load_cqr(self, filename):
        assert self.flags['cqr_ready'] == False  and self.flags['qr_ready'] == False, \
            'NARX model already exists! Create a new object to load narx.'

        # read file
        with open(filename, "rb") as file:  # Open the file in read-binary mode
            storer = pickle.load(file)

        # store data
        self.cqr = storer['cqr']
        # self.system = storer['system']

        # update flag
        self.flags.update({
            'cqr_ready': True,
            'qr_ready': True
        })

        return None


    def plot_qr_training_history(self):
        assert self.flags['qr_ready'] == True, \
            'CQR not found! Generate or load CQR model!'
        if self.cqr['type'] == 'individual':
            # plot init
            fig, ax = plt.subplots(len(self.cqr['quantiles']), 2, figsize=(24, 6 * len(self.cqr['quantiles'])))
            fig.suptitle('Individual CQR Training History')

            for i, quantile in enumerate(self.cqr['quantiles']):

                # extracting history
                training_history = self.cqr['train_history_list'][i]

                # plot 1
                ax[i, 0].plot(training_history['epochs'], training_history['learning_rate'], label='Learning Rate')
                ax[i, 0].set_xlabel('Epochs')
                ax[i, 0].set_ylabel(f'CQR (q={quantile})\nLearning Rate')
                ax[i, 0].set_yscale('log')
                ax[i, 0].grid()

                # plot 2: Training Loss plots
                ax[i, 1].plot(training_history['epochs'], training_history['training_loss'], color='blue', label='Training Loss')
                ax[i, 1].set_xlabel('Epochs')
                ax[i, 1].set_ylabel('Training Loss', color='blue')
                ax[i, 1].set_yscale('log')
                ax[i, 1].tick_params(axis='y', labelcolor='blue')
                ax[i, 1].grid()

                # plot 2: Validation Plot
                ax_n = ax[i, 1].twinx()
                ax_n.plot(training_history['epochs'], training_history['validation_loss'], color='red', label='Validation Loss')
                ax_n.tick_params(axis='y', labelcolor='red')
                ax_n.set_ylabel('Validation Loss', color='red')
                ax_n.set_yscale('log')

            # show plot
            plt.show()


        elif self.cqr['type'] == 'all':

            # plot init
            fig, ax = plt.subplots(2, figsize=(24, 6))
            fig.suptitle('All CQR Training History')

            # extracting history
            training_history = self.cqr['train_history_list'][0]

            # plot 1
            ax[0].plot(training_history['epochs'], training_history['learning_rate'], label='Learning Rate')
            ax[0].set_xlabel('Epochs')
            ax[0].set_ylabel(f'Learning Rate')
            ax[0].set_yscale('log')

            # plot 2: Training Loss plots
            ax[1].plot(training_history['epochs'], training_history['training_loss'], color='blue',
                          label='Training Loss')
            ax[1].set_xlabel('Epochs')
            ax[1].set_ylabel('Training Loss', color='blue')
            ax[1].set_yscale('log')
            ax[1].tick_params(axis='y', labelcolor='blue')
            ax[1].grid()

            # plot 2: Validation Plot
            ax_n = ax[1].twinx()
            ax_n.plot(training_history['epochs'], training_history['validation_loss'], color='red',
                      label='Validation Loss')
            ax_n.tick_params(axis='y', labelcolor='red')
            ax_n.set_ylabel('Validation Loss', color='red')
            ax_n.set_yscale('log')

        # end
        return None


    def conform_qr(self):
        assert self.flags['qr_ready'] == True, \
            'CQR not found! Train or load CQR model!'

        # calculate the surrogate model error on cqr calibration data
        self.data['cqr_calibration_errors'] = self.surrogate_error(cqr_train_inputs=self.data['cqr_calibration_inputs'],
                                                                   cqr_train_outputs=self.data[
                                                                       'cqr_calibration_outputs'])

        # storage in convenient varaibles
        Xi = self.data['cqr_calibration_inputs']
        Yi = self.data['cqr_calibration_errors']
        n_x = self.cqr['n_x']
        n_q = self.cqr['n_q']
        quantiles = self.cqr['quantiles']
        low_quantile = self.cqr['low_quantile']
        high_quantile = self.cqr['high_quantile']
        alpha = self.cqr['alpha']
        n_samples = self.data['cqr_calibration_errors'].shape[1]

        # making quantile prediction
        Xi_troch = torch.tensor(Xi.T, dtype=torch.float32)
        with torch.no_grad():
            qr_all = self.cqr['cqr_model'](Xi_troch).cpu().numpy().T

        index_high = quantiles.index(high_quantile)
        index_low = quantiles.index(low_quantile)

        # storing the values
        q_lo = qr_all[n_x * index_low: n_x + n_x * index_low, :]
        q_hi = qr_all[n_x * index_high: n_x + n_x * index_high, :]

        for j in range(n_x):

            # conformalising one state at a time
            q_lo_xn = q_lo[j,:]
            q_hi_xn = q_hi[j, :]
            Yi_xn = Yi[j,:]

            # Generating conformity scores
            Ei_xn = np.max(np.vstack([q_lo_xn - Yi_xn, Yi_xn - q_hi_xn]), axis = 0)

            # calculating the appropriate quantile
            error_quantile = (1 - alpha) * (1 + 1/n_samples)

            # Compute the quantile
            Q_xn = np.quantile(Ei_xn, q=error_quantile)

            # storage
            if j == 0:
                    Q1_alpha = Q_xn
            else:
                Q1_alpha = np.vstack([Q1_alpha, Q_xn])

        # storage
        self.cqr['Q1_alpha'] = Q1_alpha
        self.cqr['history'] = None

        # update flag
        self.flags.update({
            'cqr_ready': True
        })

        # end
        return None

    def cqr_set_initial_guess(self, states, inputs=None):

        assert self.flags['qr_ready'] == True, \
            'CQR not found! Train or load CQR model!'

        assert self.flags['cqr_ready'] == True, \
            'QR not confromalised! Conformalise QR model.'

        assert states.shape[1] == self.cqr['order'], \
            'Number of samples must be equal to the order of the NARX model!'

        assert states.shape[0] == self.cqr['n_x'], (
            'Expected number of states is: {}, but found {}'.format(self.data['n_x'], states.shape[0]))

        if self.cqr['order']>1:
            assert isinstance(inputs, np.ndarray), "If order is more than 1, then input is needed!"

            assert states.shape[1] - 1 == inputs.shape[1], \
                'Number of samples for states should exceed that of inputs by one!'

            assert inputs.shape[0] == self.cqr['n_u'], (
                'Expected number of inputs is: {}, but found {}'.format(self.data['n_u'], inputs.shape[0]))

        # set initial condition of simulator
        #self.simulator_set_initial_guess(states=states, inputs=inputs)

        state_order = self.cqr['order']
        state_samples = states.shape[1]

        # ensuring this is the current input
        # stacking states and inputs with order
        order_states = np.vstack([states[:, state_order - i - 1:state_samples - i] for i in range(state_order)])

        # if order is 2 or more, only then previous inputs are needed
        if self.cqr['order'] > 1:
            input_order = self.cqr['order'] - 1
            input_samples = inputs.shape[1]

            order_inputs = np.vstack([inputs[:, input_order - i - 1:input_samples - i] for i in range(input_order)])

            # stacking states and inputs for narx model
            initial_cond = np.vstack([order_states, order_inputs])

        else:
            initial_cond = order_states

        # store cqr initial contition
        self.cqr['x0'] = initial_cond

        # flag update
        self.flags.update({
            'cqr_initial_condition_ready': True
        })

        # end
        return None


    def cqr_make_step(self, u0):
        assert self.flags['qr_ready'], "Qunatile regressor not ready."
        assert self.flags['cqr_ready'], "Qunatile regressor not conformalised."
        assert self.flags['cqr_initial_condition_ready'], "CQR not initialised"
        assert u0.shape[0] == self.cqr['n_u'], \
            f"u0 should have have {self.cqr['n_u']} rows but instead found {u0.shape[0]}!"
        assert u0.shape[1] == 1, \
            f"u0 should have have 1 columns but instead found {u0.shape[1]}!"

        # init
        x0 = self.cqr['x0']
        n_x = self.cqr['n_x']
        n_q = self.cqr['n_q']
        n_u = self.cqr['n_u']
        order = self.cqr['order']
        Q1_alpha = self.cqr['Q1_alpha']


        # segregating states and inputs
        states = x0[0:n_x*order, :]
        inputs = x0[n_x*order:, :]

        # stacking all data
        X = np.vstack([states, u0, inputs])

        # setting default device
        self._set_device(torch_device=self.cqr['full_model'].torch_device)

        # narx_input = self.input_preprocessing(states=order_states, inputs=order_inputs)
        X_torch = torch.tensor(X.T, dtype=torch.float32)

        # making full model prediction
        with torch.no_grad():
            y_pred = self.cqr['full_model'](X_torch).cpu().numpy().T

        # reshaping from a column vector to row with states and column with different quantiles
        y_pred = self.reshape(y_pred, shape=(n_x, -1))

        # mean state prediction
        x0 = self.reshape(y_pred[:, 0], shape=(n_x, 1))
        q_alpha = y_pred[:, 1:]

        # higher quantile calculations
        q_alpha_high = self.reshape(q_alpha[:, 0], shape=(n_x, 1))

        # shifting up quantile
        error0_cqr_high = q_alpha_high + Q1_alpha
        error0_cqr_high = self.reshape(error0_cqr_high, shape=(n_x, 1))

        # lower quantile calculations
        q_alpha_low = self.reshape(q_alpha[:, 1], shape=(n_x, 1))

        # shifting down quantile
        error0_cqr_low = q_alpha_low - Q1_alpha
        error0_cqr_low = self.reshape(error0_cqr_low, shape=(n_x, 1))

        # real state predictions
        x0_cqr_high = x0 + error0_cqr_high
        x0_cqr_low = x0 + error0_cqr_low

        # pushing oldest state out of system and inserting the current state
        new_states = np.vstack([x0, states[0:(order-1)*n_x, :]])

        if order>1:

            # pushing oldest input out of system and inserting the current input
            new_inputs = np.vstack([u0, inputs[0:(order-2)*n_u, :]])

            # setting new initial guess by removing the last timestamp data
            self.cqr_set_initial_guess(states=self.reshape(new_states, shape=(n_x, -1)),
                                       inputs=self.reshape(new_inputs, shape=(n_u, -1)))

        else:
            self.cqr_set_initial_guess(states=self.reshape(new_states, shape=(n_x, -1)))

        # storing simulation history
        if self.cqr['history']==None:
            history = {}
            history['x0'] =x0
            history['x0_cqr_high'] = x0_cqr_high
            history['x0_cqr_low'] = x0_cqr_low
            history['time'] = [0.0]
            history['u0'] = u0

            self.cqr['history'] = history

        else:
            history = self.cqr['history']

            history['x0'] = np.hstack([history['x0'], x0])
            history['x0_cqr_high'] = np.hstack([history['x0_cqr_high'], x0_cqr_high])
            history['x0_cqr_low'] = np.hstack([history['x0_cqr_low'], x0_cqr_low])
            history['time'].append(history['time'][-1] + self.cqr['t_step'])
            history['u0'] = np.hstack([history['u0'], u0])

            self.cqr['history'] = history


        # return predictions
        return x0, x0_cqr_high, x0_cqr_low


    def plot_qr_error(self):

        assert self.flags['qr_ready'], "Qunatile regressor not ready."

        # calculate the surrogate model error on cqr calibration data
        self.data['cqr_calibration_errors'] = self.surrogate_error(cqr_train_inputs=self.data['cqr_calibration_inputs'],
                                                                   cqr_train_outputs=self.data[
                                                                       'cqr_calibration_outputs'])


        # init
        n_x = self.cqr['n_x']
        n_q = self.cqr['n_q']
        quantiles = self.cqr['quantiles']
        low_quantile = self.cqr['low_quantile']
        high_quantile = self.cqr['high_quantile']
        n_a = 1

        # setting default device
        self._set_device(torch_device=self.cqr['cqr_model'].torch_device)

        # narx_input = self.input_preprocessing(states=order_states, inputs=order_inputs)
        X_narx = torch.tensor(self.data['cqr_calibration_inputs'].T, dtype=torch.float32)

        # making prediction
        with torch.no_grad():
            Y_pred = self.cqr['cqr_model'](X_narx).cpu().numpy().T

        # setting up plots
        fig, ax = plt.subplots(n_x, figsize=(24, 6 * n_x))
        fig.suptitle('QR Error plots')

        # sorting with timestamps
        x = self.data['cqr_calibration_timestamps'].reshape(-1, )
        sorted_indices = np.argsort(x)  # Get indices that would sort x
        x_sorted = x[sorted_indices]

        # plot for each state
        for i in range(n_x):
            # plot the real mean
            ax[i].plot(x_sorted, self.data['cqr_calibration_errors'][i, :][sorted_indices], label=f'real mean')

            for j in range(n_q):

                index = i + n_x*j

                # plotting cqr high side
                if j<n_a:
                    ax[i].plot(x_sorted, Y_pred[index, :][sorted_indices], label=f'quantile={high_quantile}')

                # plotting cqr low side
                elif j>=n_a:
                    ax[i].plot(x_sorted, Y_pred[index, :][sorted_indices], label=f'quantile={low_quantile}')

            # extras
            ax[i].set_ylabel(f'State Error {i}')
            ax[i].legend()

        # x label
        ax[-1].set_xlabel(f'Time [s]')

        # show plot
        plt.show()

        # end
        return None


    def plot_cqr_error(self):
        assert self.flags['qr_ready'], "Qunatile regressor not ready."
        assert self.flags['cqr_ready'], "Qunatile regressor not conformalised."

        # extracting data
        X_test = self.data['test_inputs']
        Y_test = self.data['test_outputs']
        t_test = self.data['test_timestamps']

        # init
        order = self.cqr['order']
        n_x = self.cqr['n_x']
        n_u = self.cqr['n_u']
        low_quantile = self.cqr['low_quantile']
        high_quantile = self.cqr['high_quantile']
        n_samples = X_test.shape[1]
        alpha = self.cqr['alpha']



        # calculating model intervals
        for i in tqdm(range(n_samples), desc='Calculating surrogate model state intervals'):

            # extracting individual state and input histories
            states_history = X_test[0:n_x*order, i]
            inputs_n = X_test[n_x*order:, i]
            u0 = inputs_n[0:n_u]
            inputs_history = inputs_n[n_u:]

            # simulating system
            if order>1:
                self.cqr_set_initial_guess(states=self.reshape(states_history, shape=(n_x, -1)),
                                           inputs=self.reshape(inputs_history, shape=(n_u, -1)))
                x0, x0_cqr_high, x0_cqr_low = self.cqr_make_step(u0=self.reshape(u0, shape=(n_u, 1)))
            else:
                self.cqr_set_initial_guess(states=self.reshape(states_history, shape=(n_x, -1)))
                x0, x0_cqr_high, x0_cqr_low = self.cqr_make_step(u0=self.reshape(u0, shape=(n_u, 1)))

            # storage
            if i==0:
                Y_predicted_mean = x0
                Y_predicted_high = x0_cqr_high
                Y_predicted_low = x0_cqr_low

            else:
                Y_predicted_mean = np.hstack([Y_predicted_mean, x0])
                Y_predicted_high = np.hstack([Y_predicted_high, x0_cqr_high])
                Y_predicted_low = np.hstack([Y_predicted_low, x0_cqr_low])

        # generating the plots
        fig, ax = plt.subplots(n_x, figsize=(24, 6 * n_x))
        fig.suptitle('CQR State plots')

        # sorting according to timestamps
        x = t_test.reshape(-1, )
        sorted_indices = np.argsort(x)  # Get indices that would sort x
        x_sorted = x[sorted_indices]

        # plot for each state
        for i in range(n_x):

            # plot the predicted mean
            ax[i].plot(x_sorted, Y_predicted_mean[i, :][sorted_indices], label=f'predicted mean', color='blue')

            # plot the real mean
            ax[i].plot(x_sorted, Y_test[i, :][sorted_indices], label=f'real mean', color='orange')

            # plotting cqr high side
            ax[i].scatter(x_sorted, Y_predicted_high[i, :][sorted_indices], label=f'quantile={high_quantile}',
                          color='green')

            # plotting cqr low side
            ax[i].scatter(x_sorted, Y_predicted_low[i, :][sorted_indices], label=f'quantile={low_quantile}',
                          color='purple')

            # plotting the shaded region
            ax[i].fill_between(x_sorted, Y_predicted_high[i, :][sorted_indices], Y_predicted_low[i, :][sorted_indices],
                             color='grey', alpha=0.5, label=f"Confidence= {1-alpha}")

            # extras
            ax[i].set_ylabel(f'State {i}')
            ax[i].legend()

        # x label
        ax[-1].set_xlabel(f'Time [s]')

        # show plot
        plt.show()

        # end
        return None

    # Function to plot CQR error using Plotly
    def plot_cqr_error_plotly(self):
        assert self.flags['qr_ready'], "Quantile regressor not ready."
        assert self.flags['cqr_ready'], "Quantile regressor not conformalised."

        # Extracting data
        X_test = self.data['test_inputs']
        Y_test = self.data['test_outputs']
        t_test = self.data['test_timestamps']

        # Init
        order = self.cqr['order']
        n_x = self.cqr['n_x']
        n_u = self.cqr['n_u']
        low_quantile = self.cqr['low_quantile']
        high_quantile = self.cqr['high_quantile']
        n_samples = X_test.shape[1]
        alpha = self.cqr['alpha']

        # Calculating model intervals
        for i in tqdm(range(n_samples), desc='Calculating surrogate model state intervals'):
            states_history = X_test[0:n_x * order, i]
            inputs_n = X_test[n_x * order:, i]
            u0 = inputs_n[0:n_u]
            inputs_history = inputs_n[n_u:]

            if order > 1:
                self.cqr_set_initial_guess(states=self.reshape(states_history, shape=(n_x, -1)),
                                           inputs=self.reshape(inputs_history, shape=(n_u, -1)))
                x0, x0_cqr_high, x0_cqr_low = self.cqr_make_step(u0=self.reshape(u0, shape=(n_u, 1)))
            else:
                self.cqr_set_initial_guess(states=self.reshape(states_history, shape=(n_x, -1)))
                x0, x0_cqr_high, x0_cqr_low = self.cqr_make_step(u0=self.reshape(u0, shape=(n_u, 1)))

            if i == 0:
                Y_predicted_mean = x0
                Y_predicted_high = x0_cqr_high
                Y_predicted_low = x0_cqr_low
            else:
                Y_predicted_mean = np.hstack([Y_predicted_mean, x0])
                Y_predicted_high = np.hstack([Y_predicted_high, x0_cqr_high])
                Y_predicted_low = np.hstack([Y_predicted_low, x0_cqr_low])

        # Sorting according to timestamps
        x = t_test.reshape(-1, )
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]

        # Create subplots
        fig = make_subplots(rows=n_x, cols=1, shared_xaxes=True, subplot_titles=[f'State {i+1}' for i in range(n_x)])
        fig.update_layout(height=self.height_px * n_x, width=self.width_px, title_text="CQR State Plots", showlegend=True)

        # Loop through each state
        for i in range(n_x):
            # Predicted mean line (show legend for the first plot of each row)
            fig.add_trace(go.Scatter(x=x_sorted, y=Y_predicted_mean[i, sorted_indices],
                                     mode='lines', name=f'Predicted Mean',
                                     line=dict(color='blue'),
                                     showlegend=True if i == 0 else False),
                          row=i + 1, col=1)

            # Real mean line (show legend for the first plot of each row)
            fig.add_trace(go.Scatter(x=x_sorted, y=Y_test[i, sorted_indices],
                                     mode='lines', name=f'Real Mean',
                                     line=dict(color='orange'),
                                     showlegend=True if i == 0 else False),
                          row=i + 1, col=1)

            # CQR High quantile (show legend for the first plot of each row)
            fig.add_trace(go.Scatter(x=x_sorted, y=Y_predicted_high[i, sorted_indices],
                                     mode='markers', name=f'High Quantile={high_quantile}',
                                     marker=dict(color='green', size=6),
                                     showlegend=True if i == 0 else False),
                          row=i + 1, col=1)

            # CQR Low quantile (show legend for the first plot of each row)
            fig.add_trace(go.Scatter(x=x_sorted, y=Y_predicted_low[i, sorted_indices],
                                     mode='markers', name=f'Low Quantile={low_quantile}',
                                     marker=dict(color='purple', size=6),
                                     showlegend=True if i == 0 else False),
                          row=i + 1, col=1)

            # Shaded confidence interval (show legend for the first plot of each row)
            fig.add_trace(go.Scatter(x=np.concatenate((x_sorted, x_sorted[::-1])),
                                     y=np.concatenate((Y_predicted_high[i, sorted_indices],
                                                       Y_predicted_low[i, sorted_indices][::-1])),
                                     fill='toself', fillcolor='rgba(128, 128, 128, 0.5)',
                                     line=dict(color='rgba(255,255,255,0)'),
                                     name=f'Confidence {1 - alpha}',
                                     showlegend=True if i == 0 else False),
                          row=i + 1, col=1)

        # Update layout
        fig.update_xaxes(title_text="Sample [s]", row=n_x, col=1)

        # Show plot
        fig.show()
        return None

    def plot_qr_training_history_plotly(self):
        assert self.flags['qr_ready'] == True, 'CQR not found! Generate or load CQR model!'


        if self.cqr['type'] == 'individual':

            n_q = self.cqr['n_q']
            quantiles = self.cqr['quantiles']
            # Create subplots with secondary_y set in column 2
            specs = [[{}, {"secondary_y": True}] for _ in quantiles]

            # plot init
            fig = make_subplots(
                rows=n_q, cols=2,
                shared_xaxes=True,
                subplot_titles=['Learning rate', 'Loss'],
                specs=specs,
                # Enable secondary y-axis in column 2
                row_heights=[0.5] * n_q,  # Adjust row heights for better layout
                column_widths=[1] * 2
            )

            # updating layout
            fig.update_layout(title_text='Individual CQR Training History',
                              height=self.height_px * n_q, width=self.width_px)

            # making plots
            for i, quantile in enumerate(quantiles):
                # Extracting history
                training_history = self.cqr['train_history_list'][i]

                # Plot 1: Learning Rate (left column)
                fig.add_trace(go.Scatter(x=training_history['epochs'], y=training_history['learning_rate'],
                                         mode='lines', line=dict(color='red'),
                                         name='learning rate',
                                         showlegend=False),
                              row=i + 1, col=1)
                fig.update_yaxes(type='log', title_text=f'CQR (q={quantile})\nLearning Rate', row=i + 1, col=1)
                fig.update_xaxes(title_text='epochs', row=i + 1, col=1)

                # Plot 2: Training Loss (primary y-axis in right column)
                fig.add_trace(go.Scatter(x=training_history['epochs'], y=training_history['training_loss'],
                                         mode='lines', line=dict(color='green'),
                                         name='training loss',
                                         showlegend=True if i == 0 else False),
                              row=i + 1, col=2)
                fig.update_yaxes(type='log', title_text=f'CQR (q={quantile})\nTraining Loss', row=i + 1, col=2)

                # Validation Loss (secondary y-axis in right column)
                fig.add_trace(go.Scatter(x=training_history['epochs'], y=training_history['validation_loss'],
                                         mode='lines', line=dict(color='blue'),
                                         name = 'validation loss',
                                         showlegend=True if i == 0 else False),
                              row=i + 1, col=2, secondary_y=True)
                fig.update_yaxes(title_text=f'CQR (q={quantile})\nValidation Loss', type='log',
                                 row=i + 1, col=2, secondary_y=True)
                fig.update_xaxes(title_text='epochs', row=i + 1, col=2)

            fig.show()

        elif self.cqr['type'] == 'all':
            # Create subplots with secondary_y set in row 2
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                subplot_titles=['Loss History', 'Learning Rate'],
                specs=[[{"secondary_y": True}], [{"secondary_y": False}]]  # Enable secondary y-axis only in row 2
            )

            fig.update_layout(title_text='All CQR Training History', height=self.height_px, width=self.width_px)

            # Extracting history
            training_history = self.cqr['train_history_list'][0]

            # Plot 1: Training Loss (primary y-axis in row 1)
            fig.add_trace(go.Scatter(x=training_history['epochs'], y=training_history['training_loss'],
                                     mode='lines', line=dict(color='green'),
                                     name=f'training loss',
                                     showlegend=True),
                          row=1, col=1)
            fig.update_yaxes(type='log', title_text='Training Loss', row=1, col=1)

            # Validation Loss (secondary y-axis in row 1)
            fig.add_trace(go.Scatter(x=training_history['epochs'], y=training_history['validation_loss'],
                                     mode='lines', line=dict(color='red'),
                                     name=f'validation loss',
                                     showlegend=True),
                          row=1, col=1, secondary_y=True)
            fig.update_yaxes(title_text='Validation Loss', type='log', row=1, col=1, secondary_y=True)
            fig.update_xaxes(title_text='epochs', row=1, col=1)

            # Plot 2: Learning Rate (row 2)
            fig.add_trace(go.Scatter(x=training_history['epochs'], y=training_history['learning_rate'],
                                     mode='lines', line=dict(color='blue'),
                                     showlegend=False),
                          row=2, col=1)
            fig.update_yaxes(type='log', title_text='Learning Rate', row=2, col=1)
            fig.update_xaxes(title_text='epochs', row=2, col=1)

            fig.show()

        return None


    def plot_narx_training_history_plotly(self):
        assert self.flags['narx_ready'] == True, \
            'NARX not found! Generate or load NARX model!'

        # Create subplots with secondary_y set in row 2
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=['Loss History', 'Learning Rate'],
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]  # Enable secondary y-axis only in row 2
        )

        fig.update_layout(title_text='NARX Training History', height=self.height_px, width=self.width_px)

        # Extracting history
        train_history = self.narx['train_history']

        # Plot 1: Training Loss (primary y-axis in row 1)
        fig.add_trace(go.Scatter(x=train_history['epochs'], y=train_history['training_loss'],
                                 mode='lines', line=dict(color='green'),
                                 name=f'training loss',
                                 showlegend=True),
                      row=1, col=1)
        fig.update_yaxes(type='log', title_text='Training Loss', row=1, col=1)

        # Validation Loss (secondary y-axis in row 1)
        fig.add_trace(go.Scatter(x=train_history['epochs'], y=train_history['validation_loss'],
                                 mode='lines', line=dict(color='red'),
                                 name=f'validation loss',
                                 showlegend=True),
                      row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text='Validation Loss', type='log', row=1, col=1, secondary_y=True)
        fig.update_xaxes(title_text='epochs', row=1, col=1)

        # Plot 2: Learning Rate (row 2)
        fig.add_trace(go.Scatter(x=train_history['epochs'], y=train_history['learning_rate'],
                                 mode='lines', line=dict(color='blue'),
                                 showlegend=False),
                      row=2, col=1)
        fig.update_yaxes(type='log', title_text='Learning Rate', row=2, col=1)
        fig.update_xaxes(title_text='epochs', row=2, col=1)

        fig.show()

        # end
        return None


    def random_state_mpc(self, model, n_horizon, r, suppress_ipopt=True):

        # init
        n_x = self.data['n_x']
        n_u = self.data['n_u']
        order =self.data['order']
        narx_state_length = order * n_x + (order - 1) * n_u

        # states of system
        x = model.x["system_state"][0:n_x]
        x_ref = model.tvp['state_ref']

        # generating mpc class
        mpc = module.mpc_narx(model=model, order=order, n_x=n_x, n_u=n_u)

        # supperess ipopt output
        if suppress_ipopt:
            mpc.settings.supress_ipopt_output()

        # set t_step
        mpc.set_param(t_step=self.data['t_step'])

        # set horizon
        mpc.set_param(n_horizon=n_horizon)

        # setting up cost function
        mterm = sum([(x_ref[i]-x[i])**2 for i in range(n_x)])

        # passing objective function
        mpc.set_objective(mterm=mterm, lterm=0*mterm)

        # input penalisation
        mpc.set_rterm(system_input=r)

        # setting up boundaries for mpc: lower bound for states
        lbx = np.vstack([self.data['lbx'].reshape(-1,1), np.full((narx_state_length - n_x,1), -np.inf)])
        mpc.bounds['lower', '_x', 'system_state'] = lbx

        # upper bound for states
        ubx = np.vstack([self.data['ubx'].reshape(-1, 1), np.full((narx_state_length - n_x, 1), np.inf)])
        mpc.bounds['upper', '_x', 'system_state'] = ubx

        # lower bound for inputs
        mpc.bounds['lower', '_u', 'system_input'] = self.data['lbu'].reshape(-1,1)

        # upper bound for inputs
        mpc.bounds['upper', '_u', 'system_input'] = self.data['ubu'].reshape(-1,1)

        # enter random setpoints inside the box constraints
        tvp_template = mpc.get_tvp_template()

        # sending random setpoints inside box constraints
        def tvp_fun(t_ind):
            ref = np.random.uniform(self.data['lbx'], self.data['ubx']).reshape((-1,1))
            tvp_template['_tvp', :, 'state_ref'] = ref
            return tvp_template

        mpc.set_tvp_fun(tvp_fun)

        # setup
        mpc.setup()

        # storage
        self.mpc['random_state_mpc'] = mpc

        # flag update
        self.flags.update({
            'mpc_ready': True,
        })

        # end
        return mpc

    def run_simulation(self, system, iter, n_horizon, r):

        assert self.flags['qr_ready'], "Quantile regressor not ready."
        assert self.flags['cqr_ready'], "Quantile regressor not conformalised."

        # init
        narx_inputs = self.data['test_inputs']
        #narx_outputs = self.data['test_outputs']
        n_x = self.data['n_x']
        n_u = self.data['n_u']
        order = self.data['order']


        # system init
        model = system._get_model()
        simulator = system._get_simulator(model=model)
        estimator = do_mpc.estimator.StateFeedback(model= model)

        # getting controller with surrogate model inside the mpc
        surrogate_model, _ = self.narx_2_dompc()
        mpc = self.random_state_mpc(model=surrogate_model, n_horizon=n_horizon, r=r)

        # take initial guess from test data
        rnd_col = np.random.randint(narx_inputs.shape[1])  # Select a random column index

        # extracting random column
        states_history = narx_inputs[0:n_x * order, rnd_col]
        inputs_n = narx_inputs[n_x * order:, rnd_col]
        u0 = inputs_n[0:n_u]
        inputs_history = inputs_n[n_u:]

        # setting initial guess to mpc if order > 1
        if order > 1:
            # set initial guess for surrogate simulator
            #self.cqr_set_initial_guess(states=self.reshape(states_history, shape=(n_x, -1)),
            #                           inputs=self.reshape(inputs_history, shape=(n_u, -1)))
            mpc.states=self.reshape(states_history, shape=(n_x, -1))
            mpc.inputs=self.reshape(inputs_history, shape=(n_u, -1))
            mpc.narx_set_initial_guess()

        # setting initial guess to mpc if order == 1
        else:
            #self.cqr_set_initial_guess(states=self.reshape(states_history, shape=(n_x, -1)))
            mpc.states=self.reshape(states_history, shape=(n_x, -1))
            mpc.narx_set_initial_guess()

        # extracting the most recent initial state for the data
        x0 = self.reshape(states_history[0:n_x], shape=(n_x, 1))

        # setting initial guess to simulator
        simulator.x0=x0
        simulator.set_initial_guess()

        # run the main loop
        for _ in range(iter):
            u0 = mpc.narx_make_step(x0)
            #x0, x0_cqr_high, x0_cqr_low = self.cqr_make_step(u0)
            y0 = simulator.make_step(u0)
            x0 = estimator.make_step(y0)

        # storage
        self.simulation['simulator'] = simulator

        # flag update
        self.flags.update({
                'simulation_ready': True,
            })

        return None


    def plot_simulation(self):
        assert self.flags['simulation_ready'], 'Simulation not run! Run simulation first.'

        # using do-mpc for the plot
        fig, ax, graphics = do_mpc.graphics.default_plot(self.simulation['simulator'].data, figsize=(16, 9))
        graphics.plot_results()
        graphics.reset_axes()
        plt.show()

        # end
        return None






















