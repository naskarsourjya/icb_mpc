import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import do_mpc
from tqdm import tqdm
import torch
import os
import imageio
from IPython.display import display, Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


from ._graphics import plotter
from ._bmpc import MPC_Brancher
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

    def numpy_2_df(self, data_x, data_u, data_t):

        # states
        x_names = [f'state_{i + 1}' for i in range(data_x.shape[1])]
        df_x = pd.DataFrame(data_x, columns=x_names)

        # inputs
        u_names = [f'input_{i + 1}' for i in range(data_u.shape[1])]
        df_u = pd.DataFrame(data_u, columns=u_names)

        # time
        t_names = ['time']
        df_t = pd.DataFrame(data_t, columns=t_names)

        # final df
        df = pd.concat([df_t, df_x, df_u], axis=1)

        return df


    def random_input_sampler(self, system, n_samples, change_probability = 0.7):
        assert self.flags['data_stored'] == False, \
            'Data already exists! Create a new object to create new trajectory.'

        # setting up sysetm
        model = system._get_model()
        simulator = system._get_simulator(model=model)
        estimator = do_mpc.estimator.StateFeedback(model= model)

        # random initial state
        x0_init = np.random.uniform(system.lbx, system.ubx).reshape((model.n_x, 1))

        simulator.x0 = x0_init
        simulator.set_initial_guess()

        for i in tqdm(range(n_samples), desc= 'Generating data'):

            # executes if the system decides for a change
            if i==0 or np.random.rand() < change_probability:
                #u0 = np.random.uniform(system.lbu, system.ubu).reshape((-1,1))
                u0 = np.random.uniform(system.lbu, system.ubu).reshape((-1,1))
                u_prev = u0

            # executes if the system decides to not change
            else:
                u0 = u_prev

            y_next = simulator.make_step(u0)
            x0 = estimator.make_step(y_next)

        df = self.numpy_2_df(data_x= simulator.data['_x'],
                             data_u= simulator.data['_u'],
                             data_t= simulator.data['_time'])

        # storage
        self.data['simulation'] = df
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


    def input_preprocessing_deprecate(self, states, inputs):
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


    def output_preprocessing_deprecate(self, states):
        assert states.shape[0] == self.data['n_x'], (
            'Expected number of states is: {}, but found {}'.format(self.data['n_x'], states.shape[0]))

        # data gen
        narx_output = states[:,self.data['order']:]

        # end
        return narx_output


    def simulation_2_narx(self, df, order):

        # init
        df_new = df.copy()
        x_names = [f'state_{i + 1}' for i in range(self.data['n_x'])]
        u_names = [f'input_{i + 1}' for i in range(self.data['n_u'])]

        # Add lagged columns
        for i in range(-1, order):
            for col in x_names:
                col_name = f"{col}_lag_{i+1}"
                df_new[col_name] = df_new[col].shift(i)

        # Add lagged columns
        for i in range(order):
            for col in u_names:
                col_name = f"{col}_lag_{i}"
                df_new[col_name] = df_new[col].shift(i)

        # clean up
        df_new.drop(columns=x_names+u_names, inplace= True)
        df_new.dropna(inplace=True)
        df_new.reset_index(inplace=True, drop=True)

        # labels for dataset
        y_label = [f'state_{i + 1}_lag_0' for i in range(self.data['n_x'])]
        x_label=[]
        for i in range(order+1):
            for j, col in enumerate(x_names):
                if i > 0:
                    col_name = f"{col}_lag_{i}"
                    x_label.append(col_name)
        for i in range(order):
            for col in u_names:
                col_name = f"{col}_lag_{i}"
                x_label.append(col_name)

        # storage
        self.data['x_label'] = x_label
        self.data['y_label'] = y_label

        # end
        return df_new


    def data_splitter(self, order, narx_train= 0.4,
                      cqr_train= 0.4, cqr_calibration= 0.1, test = 0.1):
        assert self.flags['data_stored'] == True, \
            'Data not found! First run random_trajectory_sampler(), to generate data.'

        assert order >= 1, 'Please ensure order is an integer greater than or equal to 1!'

        assert isinstance(order, int), "Order must be an integer more than or equal to 1!"

        # store order
        self.data['order'] = order
        sets = ['narx_train', 'cqr_train', 'cqr_calibration', 'test']
        ratios = [narx_train, cqr_train, cqr_calibration, test]

        # narxing data
        df = self.simulation_2_narx(df=self.data['simulation'], order=order)

        # generating splits
        for i, set in enumerate(sets[:-1]):

            ratio = ratios[i] / sum(ratios[i:])

            # last split
            if i == len(sets) -2:
                self.data[sets[-1]], self.data[set] = train_test_split(df, test_size=ratio,
                                                                       random_state=self.set_seed)

                # resetting indices
                self.data[set].reset_index(inplace=True, drop=True)
                self.data[sets[-1]].reset_index(inplace=True, drop=True)

            # every other split
            else:
                df, self.data[set] = (train_test_split(df, test_size=ratio, random_state=self.set_seed))

                # resetting indices
                self.data[set].reset_index(inplace=True, drop=True)


        # flag update
        self.flags.update({
            'data_split': True,
        })

        # end
        return None


    def _set_device(self, torch_device):

        torch.set_default_device(torch_device)

        return None


    def narx_trainer(self, hidden_layers, batch_size, learning_rate, epochs= 1000,
                     validation_split = 0.2, scheduler_flag = True, device = 'auto', lr_threshold = 1e-8):
        assert self.flags['data_stored'] == True, \
            'Data does not exist! Generate or load data!'

        df = self.data['narx_train']
        x_train = df[self.data['x_label']]
        y_train = df[self.data['y_label']]

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

        model = self.surrogate.narx_2_dompc_model(narx=self.narx)
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


    def train_individual_qr(self, alpha, hidden_layers, device = 'auto', learning_rate= 0.1, batch_size= 32,
                  validation_split= 0.2, scheduler_flag= True, epochs = 1000, lr_threshold = 1e-8):

        assert self.flags['data_split'] == True, 'Split data not found!'
        assert 0 < alpha < 1, "All alpha must be between 0 and 1"

        # training data
        df = self.data['cqr_train']
        x_train = df[self.data['x_label']]
        y_train = df[self.data['y_label']]

        # cqr class init
        self.cqr = cqr_narx(narx= self.narx.model, alpha=alpha, n_x=self.data['n_x'], n_u=self.data['n_u'],
                       order=self.data['order'], t_step=self.data['t_step'], lbx=self.data['lbx'],
                            ubx=self.data['ubx'], device=device)

        # pushing trainer settings
        self.cqr.setup_trainer(hidden_layers=hidden_layers, learning_rate=learning_rate, batch_size=batch_size,
                          validation_split=validation_split, scheduler_flag=scheduler_flag, epochs=epochs,
                          lr_threshold=lr_threshold)

        # pushing data
        self.cqr.train_individual_qr(x_train=x_train, y_train=y_train)

        # calibration data
        df = self.data['cqr_calibration']
        x_calib = df[self.data['x_label']]
        y_calib = df[self.data['y_label']]

        # conformalising
        self.cqr.conform_qr(x_calib=x_calib, y_calib=y_calib)

        # end
        return None


    def cqr_plot_qr_error(self):

        df = self.data['cqr_calibration']

        t_calib = df['time']

        self.cqr.plot_qr_error(t_test= t_calib.to_numpy())

        return None

    def plot_cqr_error_plotly(self):

        df = self.data['cqr_calibration']

        # Extracting data
        x_test = df[self.data['x_label']]
        y_test = df[self.data['y_label']]
        t_test = df['time']

        self.cqr.plot_cqr_error(x_test= x_test, y_test=y_test, t_test=t_test)

        return None


    def step_state_mpc(self, model, n_horizon, r, cqr, tightner, confidence_cutoff,
                       setpoint, max_search, suppress_ipopt=False):

        # init
        n_x = self.data['n_x']
        n_u = self.data['n_u']
        order =self.data['order']
        narx_state_length = order * n_x + (order - 1) * n_u

        # states of system
        x = model.x.master[0:n_x]
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
        mterm = (setpoint -x[0]) ** 2
        #mterm = (0 - x[0]) ** 2

        # passing objective function
        mpc.set_objective(mterm=mterm, lterm=mterm)

        # input penalisation
        mpc.set_rterm(input_1_lag_0=r)

        # setting up boundaries for mpc: lower bound for states
        #lbx = np.vstack([self.data['lbx'].reshape(-1,1), np.full((narx_state_length - n_x,1), -np.inf)])
        mpc.bounds['lower', '_x', 'state_1_lag_1'] = self.data['lbx'][0]
        mpc.bounds['lower', '_x', 'state_2_lag_1'] = self.data['lbx'][1]

        # upper bound for states
        #ubx = np.vstack([self.data['ubx'].reshape(-1, 1), np.full((narx_state_length - n_x, 1), np.inf)])
        mpc.bounds['upper', '_x', 'state_1_lag_1'] = self.data['ubx'][0]
        mpc.bounds['upper', '_x', 'state_2_lag_1'] = self.data['ubx'][1]

        # lower bound for inputs
        mpc.bounds['lower', '_u', 'input_1_lag_0'] = self.data['lbu']

        # upper bound for inputs
        mpc.bounds['upper', '_u', 'input_1_lag_0'] = self.data['ubu']

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
        self.cqr_mpc = MPC_Brancher(mpc=mpc, cqr=cqr, confidence_cutoff=confidence_cutoff,
                                    tightner=tightner, max_search=max_search)

        # flag update
        self.flags.update({
            'cqr_mpc_ready': True,
        })

        # end
        return self.cqr_mpc

    def run_simulation(self, system, iter, n_horizon, r, tightner, rnd_samples,
                       confidence_cutoff, setpoint, max_search, store_gif=False):
        # init
        df = self.data['test']
        #narx_outputs = self.data['test_outputs']
        n_x = self.data['n_x']
        n_u = self.data['n_u']
        order = self.data['order']
        all_plots = []


        # system init
        model = system._get_model()
        simulator = system._get_simulator(model=model)
        estimator = do_mpc.estimator.StateFeedback(model= model)

        # additional config for the cqr
        self.cqr.set_config(rnd_samples=rnd_samples, confidence_cutoff=confidence_cutoff)

        # getting controller with surrogate model inside the mpc
        surrogate_model = self.narx_2_dompc()
        cqr_mpc = self.step_state_mpc(model=surrogate_model, setpoint=setpoint, n_horizon=n_horizon, r=r,
                                      cqr=self.cqr, tightner=tightner, confidence_cutoff=confidence_cutoff,
                                      max_search=max_search)


        # generate a new ic
        states_history, inputs = self._simulate_initial_guess(system=system, zero_ic = True, max_iter = 10000)
        inputs_history = inputs[1:, :]

        # setting initial guess to mpc if order > 1
        if order > 1:
            cqr_mpc.states= states_history
            cqr_mpc.inputs= inputs_history
            cqr_mpc.set_initial_guess()

        # setting initial guess to mpc if order == 1
        else:
            cqr_mpc.states= states_history
            cqr_mpc.set_initial_guess()

        # extracting the most recent initial state for the data
        x0 = states_history[0, :].reshape((n_x, 1))

        # setting initial guess to simulator
        simulator.x0=x0
        simulator.set_initial_guess()

        # run the main loop
        for i in range(iter):

            u0 = cqr_mpc.make_step(x0, enable_plots = store_gif)
            y0 = simulator.make_step(u0)
            x0 = estimator.make_step(y0)

            print(f"\n\n++++#### Simulation report ####++++")
            print(f"Time: {simulator.t0}, Iteration: {i + 1} / {iter}")
            print(f"Input: {u0}")
            print(f"Measurement: {y0}")
            print(f"State Estimate: {x0}")
            print(f"++++#### End ####++++\n\n")

            if store_gif:
                all_plots.append(cqr_mpc.plot_trials_matplotlib(show_plot=False))

        # storage
        self.simulation['simulator'] = simulator
        self.store_gif = store_gif
        self.all_plots = all_plots

        # flag update
        self.flags.update({
                'simulation_ready': True,
            })


        return None


    def _simulate_initial_guess(self, system, zero_ic = False, max_iter = 10000):

        model = system._get_model()
        simulator = system._get_simulator(model=model)

        if zero_ic:
            x0 = np.zeros((model.n_x, 1))
            u0 = np.zeros((model.n_u, 1))

            assert np.all(x0 > system.lbx) and np.all(x0 < system.ubx), 'Zero state not feasible!'
            assert np.all(u0 > system.lbu) and np.all(u0 < system.ubu), 'Zero input not feasible!'

        else:
            x0 = np.random.uniform(system.lbx, system.ubx).reshape((-1, 1))
            u0 = np.random.uniform(system.lbu, system.ubu).reshape((-1, 1))

        x0_prev = x0
        u0_prev = u0
        reversed_states = []
        reversed_inputs = []

        for i in range(self.data['order']):

            for j in range(max_iter):
                simulator.x0 = x0_prev
                simulator.set_initial_guess()

                x0 = simulator.make_step(u0=u0)

                if np.all(x0 > system.lbx) and np.all(x0 < system.ubx):
                    reversed_states.append(x0_prev.reshape(1, -1))
                    reversed_inputs.append(u0_prev.reshape(1, -1))

                    x0_prev = x0
                    u0_prev = u0
                    break

                else:
                    u0 = np.random.uniform(system.lbu, system.ubu).reshape((-1, 1))

        # reversing the data
        states = list(reversed(reversed_states))
        inputs = list(reversed(reversed_inputs))

        # end
        return np.vstack(states), np.vstack(inputs)



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

        # Clean up temp files
        for fname in image_files:
            os.remove(fname)
        os.rmdir(frame_dir)

        print(f"GIF saved as {file_name}")

        # display gif
        display(Image(filename=file_name))

        return  None

    def show_gif_matplotlib(self, gif_name="matplotlib_animation.gif", gif_path="", temp_dir="matplotlib_frames", duration=0.5):
        assert self.store_gif, "Create gif not enabled in run_simulation!"

        # init
        all_plots = self.all_plots
        i = 0
        filenames = []

        # Directory to save images
        os.makedirs(temp_dir, exist_ok=True)

        # generating images
        for plots in tqdm(all_plots, desc="Storing plots"):
            for matplots in plots:
                fig, axes = matplots
                filename = os.path.join(temp_dir, f"frame_{i:03d}.png")
                fig.savefig(filename)
                filenames.append(filename)
                plt.close(fig)  # Close the figure to save memory
                i += 1

        # Read all files and create gif
        images = [imageio.v2.imread(fname) for fname in tqdm(filenames, desc='Generating gif')]
        imageio.mimsave(gif_name, images, duration=duration)

        # Clean up temp files
        for fname in filenames:
            os.remove(fname)
        os.rmdir(temp_dir)

        print(f"GIF saved to {gif_path}")

        # display gif
        display(Image(filename=gif_name))

        return None


    def check_simulator(self, system, iter):

        real_model = system._get_model()
        real_simulator = system._get_simulator(model=real_model)

        cqr_model = self.cqr

        self.narx_2_dompc()
        surrogate_model = self.surrogate

        # extracting random initial point from test data
        # take initial guess from test data
        df = self.data['test']
        narx_inputs = df[self.data['x_label']]
        n_x = self.data['n_x']
        n_u = self.data['n_u']
        order = self.data['order']
        rnd_col = np.random.randint(narx_inputs.shape[1])  # Select a random column index

        # generate a new ic
        states_history, inputs = self._simulate_initial_guess(system=system, zero_ic=True, max_iter=10000)
        inputs_history = inputs[1:, :]

        # initial cond
        real_simulator.x0 = states_history[0, :]
        real_simulator.set_initial_guess()

        # setting initial guess to mpc if order > 1
        if order > 1:
            cqr_model.states = states_history
            cqr_model.inputs = inputs_history
            cqr_model.set_initial_guess()

            surrogate_model.states = states_history
            surrogate_model.inputs = inputs_history
            surrogate_model.set_initial_guess()

        # setting initial guess to mpc if order == 1
        else:
            cqr_model.states = states_history
            cqr_model.set_initial_guess()

            surrogate_model.states = states_history
            surrogate_model.set_initial_guess()

        # main loop
        for _ in range(iter):
            # random input inside the input boundaries
            u_ref = np.random.uniform(system.lbu, system.ubu).reshape((-1, 1))

            # simulation steps
            x0_real = real_simulator.make_step(u0=u_ref)
            x0_cqr, x0_cqr_high, x0_cqr_low = cqr_model.make_step(u0=u_ref)
            x0_surrogate = surrogate_model.make_step(u0=u_ref)

        # exporting logs
        #surrogate_model.export_log(file_name = 'Surrogate Model Log.csv')
        #cqr_model.export_log(file_name = 'CQR_NARX Model Log.csv')


        # plots
        fig = make_subplots(rows=n_x, cols=1, shared_xaxes=True)
        fig.update_layout(height=self.height_px * n_x, width=self.width_px, title_text="Simulation Comparison Plots",
                          showlegend=True)

        # plot for each state
        for i in range(n_x):

            # Shaded confidence interval (show legend for the first plot of each row)
            fig.add_trace(go.Scatter(x=np.concatenate((cqr_model.history['time'].reshape(-1, ),
                                                       cqr_model.history['time'].reshape(-1, )[::-1])),
                                     y=np.concatenate((cqr_model.history['x0_cqr_high'][:, i],
                                                       cqr_model.history['x0_cqr_low'][:, i][::-1])),
                                     fill='toself', fillcolor='rgba(128, 128, 128, 0.5)',
                                     line=dict(color='rgba(255,255,255,0)'),
                                     name=f'Confidence {1 - cqr_model.alpha}',
                                     showlegend=True if i == 0 else False),
                          row=i + 1, col=1)

            fig.add_trace(go.Scatter(x=real_simulator.data['_time'].reshape(-1,), y=real_simulator.data['_x'][:, i],
                                     mode='lines', line=dict(color='red'),
                                     name='real simulation',
                                     showlegend=True if i == 0 else False),
                          row=i + 1, col=1)

            fig.add_trace(go.Scatter(x=cqr_model.history['time'].reshape(-1,), y=cqr_model.history['x0_cqr'][:, i],
                                     mode='lines', line=dict(color='green'),
                                     name='CQR Nominal',
                                     showlegend=True if i == 0 else False),
                          row=i + 1, col=1)

            fig.add_trace(go.Scatter(x=surrogate_model.history['time'].reshape(-1,), y=surrogate_model.history['x0'][:, i],
                                     mode='lines', line=dict(color='yellow'),
                                     name='Surrogate',
                                     showlegend=True if i == 0 else False),
                          row=i + 1, col=1)


            fig.update_yaxes(title_text=f' State {i + 1}', row=i + 1, col=1)
            fig.update_xaxes(title_text='Times [s]', row=i + 1, col=1)



        # show plot
        fig.show()

        return None
