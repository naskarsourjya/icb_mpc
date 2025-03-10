import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import do_mpc
from tqdm import tqdm
import torch
import os
import imageio
from IPython.display import display, Image


from ._graphics import plotter
from ._narxwrapper import MPC_Brancher
from ._narx import narx
from ._cqr import cqr_narx
from ._surrogate import Surrogate


class DataManager(plotter):
    def __init__(self, set_seed = None):
        super(DataManager, self).__init__()

        # for repeatable results
        if set_seed is not None:
            np.random.seed(set_seed)

        # data
        self.set_seed = set_seed
        self.data = {}
        self.surrogate = {}
        self.simulation={}

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

        # flag update
        self.flags.update({
            'data_split': True,
        })

        return None


    def _set_device(self, torch_device):

        torch.set_default_device(torch_device)

        return None


    def narx_trainer(self, hidden_layers, batch_size, learning_rate, epochs= 1000,
                     validation_split = 0.2, scheduler_flag = True, device = 'auto', lr_threshold = 1e-8):
        assert self.flags['data_stored'] == True, \
            'Data does not exist! Generate or load data!'

        x_train = self.data['narx_train_inputs']
        y_train = self.data['narx_train_outputs']

        self.narx = narx(n_x=self.data['n_x'], n_u=self.data['n_u'], order= self.data['order'],
                         t_step=self.data['t_step'], set_seed=self.set_seed, device=device)

        self.narx.setup_trainer(hidden_layers=hidden_layers, batch_size=batch_size,
                                learning_rate=learning_rate, epochs=epochs, validation_split=validation_split,
                                scheduler_flag=scheduler_flag, lr_threshold=lr_threshold)

        self.narx.train(x_train=x_train, y_train=y_train)

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


    def narx_2_dompc(self):
        self.surrogate = Surrogate(n_x=self.data['n_x'], n_u=self.data['n_u'],
                                  order=self.data['order'], t_step=self.data['t_step'])

        model = self.surrogate.narx_2_dompc_model(narx=self.narx.model)
        self.surrogate.create_simulator()

        # end
        return model

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


    def surrogate_error(self, cqr_train_inputs, cqr_train_outputs):
        assert self.flags['data_split'] == True, 'Split data not found!'

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
            self.surrogate.states=states
            self.surrogate.inputs=input_history
            self.surrogate.set_initial_guess()

            # simulating system
            x0 = self.surrogate.make_step(u0= input)

            # appending error
            delta = output - x0
            error.append(delta)

        # error np array
        error = np.concatenate(error, axis=1)

        # end
        return error




    def train_individual_qr(self, alpha, hidden_layers, device = 'auto', learning_rate= 0.1, batch_size= 32,
                  validation_split= 0.2, scheduler_flag= True, epochs = 1000, lr_threshold = 1e-8):

        assert self.flags['data_split'] == True, 'Split data not found!'
        assert 0 < alpha < 1, "All alpha must be between 0 and 1"

        # calculate the surrogate model error
        self.data['cqr_train_errors'] = self.surrogate_error(cqr_train_inputs= self.data['cqr_train_inputs'],
                                                             cqr_train_outputs= self.data['cqr_train_outputs'])

        x_train = self.data['cqr_train_inputs']
        y_train = self.data['cqr_train_errors']

        # cqr class init
        self.cqr = cqr_narx(narx= self.narx.model, alpha=alpha, n_x=self.data['n_x'], n_u=self.data['n_u'],
                       order=self.data['order'], t_step=self.data['t_step'], lbx=self.data['lbx'],
                            ubx=self.data['ubx'], device=device, set_seed=self.set_seed)

        # pushing trainer settings
        self.cqr.setup_trainer(hidden_layers=hidden_layers, learning_rate=learning_rate, batch_size=batch_size,
                          validation_split=validation_split, scheduler_flag=scheduler_flag, epochs=epochs,
                          lr_threshold=lr_threshold)

        # pushing data
        self.cqr.train_individual_qr(x_train=x_train, y_train=y_train)

        # calculate the surrogate model error on cqr calibration data
        self.data['cqr_calibration_errors'] = self.surrogate_error(cqr_train_inputs=self.data['cqr_calibration_inputs'],
                                                                   cqr_train_outputs=self.data[
                                                                       'cqr_calibration_outputs'])

        # storage in convenient varaibles
        x_calib = self.data['cqr_calibration_inputs']
        y_calib = self.data['cqr_calibration_errors']

        self.cqr.conform_qr(x_calib=x_calib, y_calib=y_calib)

        # end
        return None


    def train_all_qr(self, alpha, hidden_layers, device = 'auto', learning_rate= 0.1, batch_size= 32,
                  validation_split= 0.2, scheduler_flag= True, epochs = 1000, lr_threshold = 1e-8):

        assert self.flags['data_split'] == True, 'Split data not found!'
        assert 0 < alpha < 1, "All alpha must be between 0 and 1"

        # calculate the surrogate model error on cqr training data
        self.data['cqr_train_errors'] = self.surrogate_error(cqr_train_inputs= self.data['cqr_train_inputs'],
                                                             cqr_train_outputs= self.data['cqr_train_outputs'])

        x_train = self.data['cqr_train_inputs']
        y_train = self.data['cqr_train_errors']

        # cqr class init
        self.cqr = cqr_narx(narx=self.narx.model, alpha=alpha, n_x=self.data['n_x'], n_u=self.data['n_u'],
                            order=self.data['order'], t_step=self.data['t_step'], lbx=self.data['lbx'],
                            ubx=self.data['ubx'], device=device, set_seed=self.set_seed)

        # pushing trainer settings
        self.cqr.setup_trainer(hidden_layers=hidden_layers, learning_rate=learning_rate, batch_size=batch_size,
                               validation_split=validation_split, scheduler_flag=scheduler_flag, epochs=epochs,
                               lr_threshold=lr_threshold)

        # pushing data
        self.cqr.train_all_qr(x_train=x_train, y_train=y_train)

        # calculate the surrogate model error on cqr calibration data
        self.data['cqr_calibration_errors'] = self.surrogate_error(cqr_train_inputs=self.data['cqr_calibration_inputs'],
                                                                   cqr_train_outputs=self.data[
                                                                       'cqr_calibration_outputs'])

        # storage in convenient varaibles
        x_calib = self.data['cqr_calibration_inputs']
        y_calib = self.data['cqr_calibration_errors']

        self.cqr.conform_qr(x_calib=x_calib, y_calib=y_calib)

        # end
        return None

    def cqr_plot_qr_error(self):

        x_calib = self.data['cqr_calibration_inputs']
        y_calib = self.data['cqr_calibration_errors']
        t_calib = self.data['cqr_calibration_timestamps']

        self.cqr.plot_qr_error(x_test=x_calib, y_test=y_calib, t_test= t_calib)

        return None

    def plot_cqr_error_plotly(self):

        # Extracting data
        x_test = self.data['test_inputs']
        y_test = self.data['test_outputs']
        t_test = self.data['test_timestamps']

        self.cqr.plot_cqr_error(x_test= x_test, y_test=y_test, t_test=t_test)

        return None


    def step_state_mpc(self, model, n_horizon, r, cqr, iter, step_mag=0.01, suppress_ipopt=False):

        # init
        n_x = self.data['n_x']
        n_u = self.data['n_u']
        order =self.data['order']
        narx_state_length = order * n_x + (order - 1) * n_u

        # states of system
        x = model.x["system_state"][0:n_x]
        x_ref = model.tvp['state_ref']

        # generating mpc class
        mpc = do_mpc.controller.MPC(model=model)

        # supperess ipopt output
        if suppress_ipopt:
            mpc.settings.supress_ipopt_output()

        # set t_step
        mpc.set_param(t_step=self.data['t_step'])

        # set horizon
        mpc.set_param(n_horizon=n_horizon)

        # setting up cost function
        #mterm = sum([(x_ref[i]-x[i])**2 for i in range(n_x)])
        #mterm = (x_ref[0] - x[0]) ** 2 # tracking only the  first state
        mterm = (0.5 + x[0]) ** 2

        # passing objective function
        mpc.set_objective(mterm=mterm, lterm=mterm)

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
            #range_x = self.data['ubx'] - self.data['lbx']
            tvp_template['_tvp', :, 'state_ref'] = np.array([[-0.8], [0]])
            #if t_ind<self.data['t_step']*iter/2:
            #    tvp_template['_tvp', :, 'state_ref'] = self.data['lbx'] + step_mag * range_x

            #else:
            #    tvp_template['_tvp', :, 'state_ref'] = self.data['lbx'] + (1-step_mag) * range_x

            return tvp_template

        mpc.set_tvp_fun(tvp_fun)

        # setup
        mpc.setup()

        # storage
        self.cqr_mpc = MPC_Brancher(mpc=mpc, cqr=cqr)

        # flag update
        self.flags.update({
            'cqr_mpc_ready': True,
        })

        # end
        return self.cqr_mpc

    def run_simulation(self, system, iter, n_horizon, r, store_gif=False):
        # init
        narx_inputs = self.data['test_inputs']
        #narx_outputs = self.data['test_outputs']
        n_x = self.data['n_x']
        n_u = self.data['n_u']
        order = self.data['order']
        all_plots = []


        # system init
        model = system._get_model()
        simulator = system._get_simulator(model=model)
        estimator = do_mpc.estimator.StateFeedback(model= model)

        # getting controller with surrogate model inside the mpc
        surrogate_model = self.narx_2_dompc()
        cqr_mpc = self.step_state_mpc(model=surrogate_model, n_horizon=n_horizon, r=r, cqr=self.cqr, iter=iter)

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
            cqr_mpc.states=self.reshape(states_history, shape=(n_x, -1))
            cqr_mpc.inputs=self.reshape(inputs_history, shape=(n_u, -1))
            cqr_mpc.set_initial_guess()

        # setting initial guess to mpc if order == 1
        else:
            #self.cqr_set_initial_guess(states=self.reshape(states_history, shape=(n_x, -1)))
            cqr_mpc.states=self.reshape(states_history, shape=(n_x, -1))
            cqr_mpc.set_initial_guess()

        # extracting the most recent initial state for the data
        x0 = self.reshape(states_history[0:n_x], shape=(n_x, 1))

        # setting initial guess to simulator
        simulator.x0=x0
        simulator.set_initial_guess()

        # run the main loop
        #for _ in tqdm(range(iter), desc=f'Simulating system'):
        #for _ in tqdm(range(iter), desc='Simulating'):
        for _ in range(iter):

            u0 = cqr_mpc.make_step(x0, max_iter=5, enable_plots = True)
            #x0, x0_cqr_high, x0_cqr_low = self.cqr_make_step(u0)
            y0 = simulator.make_step(u0)
            x0 = estimator.make_step(y0)

            if store_gif:
                all_plots.append(cqr_mpc.plot_trials(show_plot=False))

        # storage
        self.simulation['simulator'] = simulator
        self.store_gif = store_gif
        self.all_plots = all_plots

        # flag update
        self.flags.update({
                'simulation_ready': True,
            })


        return None


    def show_gif(self, file_name= "plotly_animation.gif", frame_dir = "plotly_frames"):
        assert self.store_gif, "Create gif not enabled in run_simulation!"

        # init
        all_plots = self.all_plots
        i = 0
        image_files = []

        # Directory to save images
        os.makedirs(frame_dir, exist_ok=True)

        # generating images
        for plots in tqdm(all_plots, desc="Storing plots"):
            for fig in plots:
                # convert to png
                img_path = f"{frame_dir}/frame_{i}.png"
                fig.write_image(img_path)
                image_files.append(img_path)
                i += 1

        # Convert images to GIF
        with imageio.get_writer(file_name, mode='I', duration=0.5) as writer:
            for img in tqdm(image_files, desc='Generating gif'):
                image = imageio.imread(img)
                writer.append_data(image)

        print(f"GIF saved as {file_name}")

        # display gif
        display(Image(filename=file_name))

        return  None
