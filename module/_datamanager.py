import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import do_mpc
from tqdm import tqdm
import torch
import os
import imageio
#from IPython.display import display, Image
from IPython.display import display as dispp
from IPython.display import Image as immm

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import time
from datetime import datetime
import scienceplots
from PIL import Image


from ._graphics import plotter
from ._icb_mpc import ICB_MPC
from ._icb_mpc_midterm import MPC_Brancher_midterm
from ._mpc_narx import MPC_NARX
from ._narx import narx
from ._cqr import cqr_narx
from ._surrogate import Surrogate
from ._cqr2dompc import Robust_Model

# plot init
plt.style.use(['science','no-latex'])

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
        self.data['sampler'] = simulator
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

            if i==-1:
                for col in x_names:
                    col_name = f"{col}_next"
                    df_new[col_name] = df_new[col].shift(i)
            else:
                for col in x_names:
                    col_name = f"{col}_lag_{i}"
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
        y_label = [f'state_{i + 1}_next' for i in range(self.data['n_x'])]
        x_label=[]
        for i in range(order):
            for j, col in enumerate(x_names):
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


    def train_narx(self, hidden_layers, batch_size, learning_rate, epochs= 1000,
                     validation_split = 0.2, scheduler_flag = True, device = 'auto', lr_threshold = 1e-8,
                   train_threshold=None):
        assert self.flags['data_stored'] == True, \
            'Data does not exist! Generate or load data!'

        df = self.data['narx_train']
        x_train = df[self.data['x_label']]
        y_train = df[self.data['y_label']]

        self.narx = narx(n_x=self.data['n_x'], n_u=self.data['n_u'], order= self.data['order'],
                         t_step=self.data['t_step'], set_seed=self.set_seed, device=device)

        self.narx.setup_trainer(hidden_layers=hidden_layers, batch_size=batch_size,
                                learning_rate=learning_rate, epochs=epochs, validation_split=validation_split,
                                scheduler_flag=scheduler_flag, lr_threshold=lr_threshold,
                                train_threshold=train_threshold)

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


    def train_cqr(self, alpha, hidden_layers, device = 'auto', learning_rate= 0.1, batch_size= 32,
                  validation_split= 0.2, scheduler_flag= True, epochs = 1000, lr_threshold = 1e-8,
                  train_threshold = None):

        assert self.flags['data_split'] == True, 'Split data not found!'
        assert 0 < alpha < 1, "All alpha must be between 0 and 1"

        # training data
        df = self.data['cqr_train']
        x_train = df[self.data['x_label']]
        y_train = df[self.data['y_label']]

        # cqr class init
        self.cqr = cqr_narx(narx= self.narx, alpha=alpha, n_x=self.data['n_x'], n_u=self.data['n_u'],
                       order=self.data['order'], t_step=self.data['t_step'], lbx=self.data['lbx'],
                            ubx=self.data['ubx'], lbu=self.data['lbu'], ubu=self.data['ubu'],  device=device)

        # pushing trainer settings
        self.cqr.setup_trainer(hidden_layers=hidden_layers, learning_rate=learning_rate, batch_size=batch_size,
                          validation_split=validation_split, scheduler_flag=scheduler_flag, epochs=epochs,
                          lr_threshold=lr_threshold, train_threshold=train_threshold)

        # pushing data
        self.cqr.train(x_train=x_train, y_train=y_train)

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


    def _simulate_initial_guess(self, system, zero_ic = False, max_iter = 10000000):

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
        reversed_states = [x0.reshape(1, -1)]
        reversed_inputs = [u0.reshape(1, -1)]

        for i in range(self.data['order']-1):

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

        assert np.vstack(states).shape == (self.data['order'], self.data['n_x']), 'IC not found!'
        assert np.vstack(inputs).shape == (self.data['order'], self.data['n_u']), 'IC not found!'

        # end
        return np.vstack(states), np.vstack(inputs)

    def show_gif_matplotlib(self, system, gif_name="matplotlib_animation.gif", gif_path="", temp_dir="matplotlib_frames",
                            duration=0.5, figsize_w= 12, figsize_h=8):
        assert self.store_gif, "Create gif not enabled in run_simulation!"

        # init
        model = system._get_model()
        all_y_labels = model.x.keys() + model.u.keys()[1:] + model.aux.keys()[1:]
        all_plots = self.all_plots
        i = 0
        filenames = []

        # Directory to save images
        os.makedirs(temp_dir, exist_ok=True)

        # generating images
        for plots in tqdm(all_plots, desc="Storing plots"):
            for matplots in plots:
                fig, axes = matplots
                fig.set_size_inches(figsize_w, figsize_h)

                for jj, ax in enumerate(axes):
                    ax.set_ylabel(all_y_labels[jj])

                filename = os.path.join(temp_dir, f"frame_{i:03d}.png")
                fig.savefig(filename)#, dpi=dpi)#, bbox_inches='tight')
                filenames.append(filename)
                plt.close(fig)  # Close the figure to save memory
                i += 1

        # Read all files and create gif
        images = [imageio.v2.imread(fname) for fname in tqdm(filenames, desc='Generating gif')]
        # fix
        standard_shape = images[0].shape
        images = [image if image.shape == standard_shape else
                  np.array(Image.fromarray(image).resize((standard_shape[1], standard_shape[0])))
                  for image in images]


        imageio.mimsave(gif_name, images, duration=duration)

        # Clean up temp files
        for fname in filenames:
            os.remove(fname)
        os.rmdir(temp_dir)

        print(f"GIF saved to {gif_path}")

        # display gif
        dispp(immm(filename=gif_name))

        return None


    def check_simulator(self, system, iter, x_init=None, u_init=None):

        # init
        order = self.data['order']
        n_x = self.data['n_x']

        # real model and simulator
        real_model = system._get_model()
        real_simulator = system._get_simulator(model=real_model)

        # extracting the pre trained cqr model
        cqr_model = self.cqr

        self.narx_2_dompc()
        surrogate_model = self.surrogate

        # random initial point is extracted if initial state not provided
        if x_init is None:

            # generate a new ic
            states_history, inputs = self._simulate_initial_guess(system=system, zero_ic=False, max_iter=10000)
            inputs_history = inputs[0:-1, :]

            # saving the new initial conditions
            x_init = states_history
            u_init = inputs_history

        # initial cond for the real simulator
        real_simulator.x0 = x_init[0, :]
        real_simulator.set_initial_guess()

        # setting initial guess to mpc if order > 1
        if order > 1:
            cqr_model.states = x_init
            cqr_model.inputs = u_init
            cqr_model.set_initial_guess()

            surrogate_model.states = x_init
            surrogate_model.inputs = u_init
            surrogate_model.set_initial_guess()

        # setting initial guess to mpc if order == 1
        else:
            cqr_model.states = x_init
            cqr_model.set_initial_guess()

            surrogate_model.states = x_init
            surrogate_model.set_initial_guess()

        # main loop
        for _ in range(iter):
            # random input inside the input boundaries
            u_ref = np.random.uniform(system.lbu, system.ubu).reshape((-1, 1))

            # simulation steps
            x0_real = real_simulator.make_step(u0=u_ref)
            x0_cqr, x0_cqr_high, x0_cqr_low = cqr_model.make_step(u0=u_ref.reshape((1, -1)))
            x0_surrogate = surrogate_model.make_step(u0=u_ref.reshape((1, -1)))

        # exporting logs
        #surrogate_model.export_log(file_name = 'Surrogate Model Log.csv')
        #cqr_model.export_log(file_name = 'CQR_NARX Model Log.csv')


        # plots
        fig = make_subplots(rows=n_x, cols=1, shared_xaxes=True)
        fig.update_layout(height=self.height_px * 100, width=self.width_px * 100, title_text="Simulation Comparison Plots",
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


    def check_simulator_mpc(self, system, iter, setpoint, n_horizon, r, x_init=None, u_init=None):

        # init
        n_x = self.data['n_x']
        n_u = self.data['n_u']
        order = self.data['order']
        u0_list = []

        # real model and simulator
        real_model = system._get_model()
        real_simulator = system._get_simulator(model=real_model)

        # extracting the pretrained cqr
        cqr_model = self.cqr

        # generating the surrogate model
        self.narx_2_dompc()
        surrogate_model = self.surrogate

        # creating the mpc class with the surrogate model
        surrogate_mpc = system._get_surrogate_mpc(surrogate_model=self.surrogate.model, n_x=n_x,
                                                  n_u=n_u, setpoint=setpoint, n_horizon=n_horizon, r=r)

        # wrapper class for mpc to handle NARX orders
        mpc_sur = MPC_NARX(mpc=surrogate_mpc, n_x= n_x, n_u=n_u, order=order)

        # random initial point is extracted if initial state not provided
        if x_init is None:
            # generate a new ic
            states_history, inputs = self._simulate_initial_guess(system=system, zero_ic=False, max_iter=10000)
            inputs_history = inputs[0:-1, :]

            # saving the new initial conditions
            x_init = states_history
            u_init = inputs_history

        # initial cond
        x0_real = x_init[0, :].reshape((-1, 1))
        real_simulator.x0 = x0_real
        real_simulator.set_initial_guess()
        #real_mpc.set_initial_guess()

        # setting initial guess to mpc if order > 1
        if order > 1:
            cqr_model.states = x_init
            cqr_model.inputs = u_init
            cqr_model.set_initial_guess()

            surrogate_model.states = x_init
            surrogate_model.inputs = u_init
            surrogate_model.set_initial_guess()

            mpc_sur.states = x_init
            mpc_sur.inputs = u_init
            mpc_sur.set_initial_guess()

        # setting initial guess to mpc if order == 1
        else:
            cqr_model.states = x_init
            cqr_model.set_initial_guess()

            surrogate_model.states = x_init
            surrogate_model.set_initial_guess()

            mpc_sur.states = x_init
            mpc_sur.set_initial_guess()

        # main loop
        for _ in range(iter):
            # random input inside the input boundaries
            u0_surrogate = mpc_sur.make_step(x0=x0_real)
            #u0_real = real_mpc.make_step(x0= x0_real)

            # simulation steps
            x0_real = real_simulator.make_step(u0=u0_surrogate)
            #x0_cqr, x0_cqr_high, x0_cqr_low = cqr_model.make_step(u0=u0_surrogate.reshape((1, -1)))
            #x0_surrogate = surrogate_model.make_step(u0=u0_surrogate.reshape((1, -1)))

            # storage
            u0_list.append(u0_surrogate.reshape((1, -1)))

        # exporting logs
        #surrogate_model.export_log(file_name = 'Surrogate Model Log.csv')
        #cqr_model.export_log(file_name = 'CQR_NARX Model Log.csv')
        u0_list_numpy = np.vstack(u0_list)

        fig, axes = plt.subplots(nrows=n_x + n_u, ncols=1, sharex=True,
                                 figsize=(self.width_px, self.height_px))
        if (n_x + n_u) == 1:
            axes = [axes]  # Ensure axes is iterable

        # Plot states
        for i in range(n_x):
            ax = axes[i]
            time = cqr_model.history['time'].reshape(-1, )

            # Confidence interval
            upper = cqr_model.history['x0_cqr_high'][:, i]
            lower = cqr_model.history['x0_cqr_low'][:, i]
            ax.fill_between(time, lower, upper, color='gray', alpha=0.5,
                            label=f'Confidence {1 - cqr_model.alpha}' if i == 0 else "")

            # Real simulation
            ax.plot(real_simulator.data['_time'], real_simulator.data['_x'][:, i], color='red',
                    label='Real Simulation' if i == 0 else "")

            # CQR Nominal
            #ax.plot(time, cqr_model.history['x0_cqr'][:, i], color='green', label='CQR Nominal' if i == 0 else "")

            # Surrogate
            #ax.plot(surrogate_model.history['time'], surrogate_model.history['x0'][:, i], color='yellow',
            #        label='Surrogate' if i == 0 else "")

            # System bounds (upper and lower)
            ax.plot(real_simulator.data['_time'], [self.cqr.ubx[i]] * real_simulator.data['_time'].shape[0], color='grey', linestyle='solid',
                    label='System Bounds' if i == 0 else None)
            ax.plot(real_simulator.data['_time'], [self.cqr.lbx[i]] * real_simulator.data['_time'].shape[0], color='grey', linestyle='solid')


            ax.set_ylabel(f'State {i + 1}')
            ax.grid()
            if i == 0:
                ax.legend(loc='upper right')

        # Plot inputs
        for j in range(n_u):
            ax = axes[n_x + j]
            ax.plot(real_simulator.data['_time'], u0_list_numpy[:, j], color='black',
                    label='Real Simulation' if j == 0 else "")
            ax.set_ylabel(f'Input {j + 1}')
            ax.set_xlabel('Time [s]')
            ax.grid()
            if j == 0:
                ax.legend(loc='upper right')

        plt.suptitle("Simulation Comparison Plots")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        return None


    def case_study_1(self, system, iter, n_horizon, r, tightner, rnd_samples,
                       confidence_cutoff, setpoint, max_search, R, Q, store_gif=False, x_init=None, u_init=None):
        # init
        df = self.data['test']
        #narx_outputs = self.data['test_outputs']
        n_x = self.data['n_x']
        n_u = self.data['n_u']
        order = self.data['order']
        all_plots = []
        closed_loop_cost = []
        opt_success = []
        ic_flag = False if x_init is None else True

        # this is very specific to the mpc class for closed loop calc
        r_mpc = r * np.array([[0.1, 0],
                              [0,   1e-3]])
        u_prev = np.array([[0], [0]])

        # system init
        model = system._get_model()
        simulator = system._get_simulator(model=model)
        estimator = do_mpc.estimator.StateFeedback(model= model)

        # additional config for the cqr
        self.cqr.set_config(rnd_samples=rnd_samples, confidence_cutoff=confidence_cutoff)

        # getting controller with surrogate model inside the mpc
        surrogate_model = self.narx_2_dompc()

        # creating the mpc class with the surrogate model
        surrogate_mpc = system._get_surrogate_mpc(surrogate_model=surrogate_model, n_x=n_x,
                                                  n_u=n_u, setpoint=setpoint, n_horizon=n_horizon, r=r)

        # storage
        cqr_mpc = ICB_MPC(mpc=surrogate_mpc, cqr=self.cqr, confidence_cutoff=confidence_cutoff,
                                    tightner=tightner, R=R, Q=Q, max_search=max_search)

        # flag update
        self.flags.update({
            'cqr_mpc_ready': True,
        })

        # generate a new ic
        if x_init is None:
            states_history, inputs = self._simulate_initial_guess(system=system, zero_ic = False, max_iter = 10000)
            inputs_history = inputs[1:, :]

            x_init = states_history
            u_init = inputs_history

        # setting initial guess to mpc if order > 1
        if order > 1:
            cqr_mpc.states= x_init
            cqr_mpc.inputs= u_init
            cqr_mpc.set_initial_guess()

        # setting initial guess to mpc if order == 1
        else:
            cqr_mpc.states= x_init.reshape((-1, self.data['n_x']))
            cqr_mpc.set_initial_guess()

        # extracting the most recent initial state for the data
        x0 = x_init[0, :].reshape((n_x, 1))

        # setting initial guess to simulator
        simulator.x0=x0
        simulator.set_initial_guess()

        start_time = time.time()

        # run the main loop
        for i in range(iter):

            u0 = cqr_mpc.make_step(x0, enable_plots = store_gif)
            y0 = simulator.make_step(u0)
            x0 = estimator.make_step(y0)

            # calculating closed loop cost
            cl_cost = -x0[1,0] + (u0-u_prev).T @ r_mpc @ (u0-u_prev)
            u_prev = u0

            #closed_loop_cost.append(cqr_mpc.solver_stats['iterations']['obj'][-1])
            closed_loop_cost.append(cl_cost)
            opt_success.append(cqr_mpc.solver_stats['success'])

            print(f"\n\n++++#### Simulation report ####++++")
            print(f"Time: {simulator.t0}, Iteration: {i + 1} / {iter}")
            print(f"Input: {u0}")
            print(f"Measurement: {y0}")
            print(f"State Estimate: {x0}")
            print(f"++++#### End ####++++\n\n")

            if store_gif:
                all_plots.append(cqr_mpc.plot_trials_matplotlib(show_plot=False))

        end_time = time.time()

        violation_flag = self._check_boundary__violation(system=system, simulator=simulator)

        # storage
        self.simulation['simulator'] = simulator
        self.simulation['cs1'] = simulator
        self.store_gif = store_gif
        self.all_plots = all_plots

        # flag update
        self.flags.update({
                'simulation_ready': True,
            })

        # pass through
        self.cs1_results = {'Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'Type': 'cs1',
                            'No Samples': self.data['n_samples'],
                            'Surrogate Nodes': sum(self.narx.hidden_layers),
                            'CQR Nodes': sum(self.cqr.hidden_layers),
                            'CQR Alpha': self.cqr.alpha,
                            'Initial Condition': ic_flag,
                            'Optimizer Success': all(opt_success),
                            'Closed Loop Cost': sum(closed_loop_cost) / len(closed_loop_cost),
                            'Boundary Violation': violation_flag,
                            'Iterations': iter,
                            'MPC Horizon': n_horizon,
                            'MPC Input Cost': r,
                            'ICBMPC Tightner': tightner,
                            'ICBMPC Random Sampling':rnd_samples,
                            'ICBMPC Confidence Cutoff': confidence_cutoff,
                            'MPC Setpoint': setpoint,
                            'ICBMPC Max Search':max_search,
                            'Time': end_time-start_time}
        #(self, system, iter, n_horizon, r, tightner, rnd_samples, confidence_cutoff, setpoint, max_search, R, Q, store_gif=False, x_init=None, u_init=None)
        # end
        return self.cs1_results


    def _check_boundary__violation(self, system, simulator):

        n_sampes = len(simulator.data['_time'])
        ubx_stacked = np.vstack([system.ubx]*n_sampes)
        lbx_stacked = np.vstack([system.lbx] * n_sampes)
        ubu_stacked = np.vstack([system.ubu] * n_sampes)
        lbu_stacked = np.vstack([system.lbu] * n_sampes)

        if np.all(simulator.data['_x']>=lbx_stacked) and np.all(simulator.data['_x']<=ubx_stacked) and np.all(simulator.data['_u']>=lbu_stacked) and np.all(simulator.data['_u']<=ubu_stacked):
            return False
        else:
            return True



    def setup_case_study_2(self, hidden_layers, system, setpoint, n_horizon, r, epochs= 1000, batch_size=1000,
                     learning_rate=0.1, validation_split=0.2,
                     scheduler_flag= True, device='auto', lr_threshold = 1e-8, train_threshold= None):

        assert self.flags['data_stored'] == True, \
            'Data does not exist! Generate or load data!'

        # init
        df_names = ['narx_train', 'cqr_train', 'cqr_calibration']
        df_list = []

        # sxtracting all datasets
        for dataset_name in df_names:
            df = self.data[dataset_name]
            df_list.append(df)

        # combine all dataframes
        combined_df = pd.concat(df_list, ignore_index=True)

        # extract training data
        x_train = combined_df[self.data['x_label']]
        y_train = combined_df[self.data['y_label']]

        # init narx model
        narx_model = narx(n_x=self.data['n_x'], n_u=self.data['n_u'], order=self.data['order'],
                         t_step=self.data['t_step'], set_seed=self.set_seed, device=device)

        # setup training parameters
        narx_model.setup_trainer(hidden_layers=hidden_layers, batch_size=batch_size,
                                learning_rate=learning_rate, epochs=epochs, validation_split=validation_split,
                                scheduler_flag=scheduler_flag, lr_threshold=lr_threshold,
                                train_threshold=train_threshold)

        # train model
        narx_model.train(x_train=x_train, y_train=y_train)

        # plot the training performance
        narx_model.plot_narx_training_history()

        # init surrogate model
        surrogate_model = Surrogate(n_x=self.data['n_x'], n_u=self.data['n_u'],
                                   order=self.data['order'], t_step=self.data['t_step'])

        # create dompc surrogate model
        surrogate_model_dompc = surrogate_model.narx_2_dompc_model(narx=narx_model)

        # creating the mpc class with the surrogate model
        surrogate_mpc = system._get_surrogate_mpc(surrogate_model=surrogate_model_dompc, n_x=self.data['n_x'],
                                                  n_u=self.data['n_u'], setpoint=setpoint, n_horizon=n_horizon, r=r)

        # wrapper class for mpc to handle NARX orders
        self.cs2_surrogate = MPC_NARX(mpc=surrogate_mpc, n_x=self.data['n_x'], n_u=self.data['n_u'], order=self.data['order'])

        # preformance
        self.cs2_results = {'Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'Type': 'cs2',
                            'No Samples': self.data['n_samples'],
                            'Surrogate Nodes': sum(narx_model.hidden_layers),
                            'MPC Horizon': n_horizon,
                            'MPC Input Cost': r,
                            'MPC Setpoint': setpoint}

        # end
        return None


    def case_study_2(self, system, iter, x_init=None, u_init=None):

        # init
        order = self.data['order']
        mpc_sur = self.cs2_surrogate
        closed_loop_cost = []
        opt_success = []
        u_prev = np.array([[0], [0]])
        r_mpc = self.cs2_results['MPC Input Cost'] * np.array([[0.1, 0],
                                                           [0,   1e-3]])
        ic_flag = False if x_init is None else True


        # real model and simulator
        real_model = system._get_model()
        real_simulator = system._get_simulator(model=real_model)

        # random initial point is extracted if initial state not provided
        if x_init is None:
            # generate a new ic
            states_history, inputs = self._simulate_initial_guess(system=system, zero_ic=False, max_iter=10000)
            inputs_history = inputs[0:-1, :]

            # saving the new initial conditions
            x_init = states_history
            u_init = inputs_history

        # initial cond
        x0_real = x_init[0, :].reshape((-1, 1))
        real_simulator.x0 = x0_real
        real_simulator.set_initial_guess()

        # setting initial guess to mpc if order > 1
        if order > 1:
            mpc_sur.states = x_init
            mpc_sur.inputs = u_init
            mpc_sur.set_initial_guess()

        # setting initial guess to mpc if order == 1
        else:
            mpc_sur.states = x_init
            mpc_sur.set_initial_guess()

        start_time = time.time()

        # main loop
        for _ in range(iter):

            # random input inside the input boundaries
            u0_surrogate = mpc_sur.make_step(x0=x0_real)

            # simulation steps
            x0_real = real_simulator.make_step(u0=u0_surrogate)

            # calculating closed loop cost
            cl_cost = -x0_real[1,0] + (u0_surrogate-u_prev).T @ r_mpc @ (u0_surrogate-u_prev)
            u_prev = u0_surrogate

            #closed_loop_cost.append(cqr_mpc.solver_stats['iterations']['obj'][-1])
            closed_loop_cost.append(cl_cost)
            opt_success.append(mpc_sur.mpc.solver_stats['success'])

        end_time = time.time()

        violation_flag = self._check_boundary__violation(system=system, simulator=real_simulator)

        # storage
        self.simulation['simulator'] = real_simulator
        self.simulation['cs2'] = real_simulator

        # flag update
        self.flags.update({
            'simulation_ready': True,
        })

        # pass through
        self.cs2_results['Optimizer Success'] = all(opt_success)
        self.cs2_results['Closed Loop Cost'] = sum(closed_loop_cost) / len(closed_loop_cost)
        self.cs2_results['Boundary Violation'] = violation_flag
        self.cs2_results['Iterations'] = iter
        self.cs2_results['Time'] = end_time-start_time
        self.cs2_results['Initial Condition'] = ic_flag


        # end
        return self.cs2_results



    def setup_case_study_3(self, system, n_horizon, r_horizon, r, setpoint):

        # init
        robust_model_var = Robust_Model()

        # convert cqr to a robust do_mpc model
        robust_model = robust_model_var.convert2dompc(cqr=self.cqr)

        robust_mpc = system._get_robust_mpc(robust_model=robust_model, n_x=self.data['n_x'], n_u=self.data['n_u'],
                                            n_horizon= n_horizon, r_horizon=r_horizon, r=r, setpoint=setpoint)

        self.robust_mpc_narx = MPC_NARX(mpc=robust_mpc, n_x=self.data['n_x'], n_u=self.data['n_u'],
                                        order=self.data['order'])

        # preformance
        self.cs3_results = {'Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'Type': 'cs3',
                            'No Samples': self.data['n_samples'],
                            'Surrogate Nodes': sum(self.narx.hidden_layers),
                            'CQR Nodes': sum(self.cqr.hidden_layers),
                            'CQR Alpha': self.cqr.alpha,
                            'MPC Horizon': n_horizon,
                            'MPC Input Cost': r,
                            'MPC Setpoint': setpoint}

        return None


    def case_study_3(self, system, iter, x_init=None, u_init=None):

        # init
        n_x = self.data['n_x']
        n_u = self.data['n_u']
        order = self.data['order']
        robust_mpc_narx = self.robust_mpc_narx
        closed_loop_cost = []
        opt_success = []
        r_mpc = self.cs3_results['MPC Input Cost'] * np.array([[0.1, 0],
                                                           [0,   1e-3]])
        u_prev = np.array([[0], [0]])
        ic_flag = False if x_init is None else True

        # real model and simulator
        real_model = system._get_model()
        real_simulator = system._get_simulator(model=real_model)

        # random initial point is extracted if initial state not provided
        if x_init is None:
            # generate a new ic
            states_history, inputs = self._simulate_initial_guess(system=system, zero_ic=False, max_iter=10000)
            inputs_history = inputs[0:-1, :]

            # saving the new initial conditions
            x_init = states_history
            u_init = inputs_history

        # initial cond
        x0_real = x_init[0, :].reshape((-1, 1))
        real_simulator.x0 = x0_real
        real_simulator.set_initial_guess()

        # setting initial guess to mpc if order > 1
        if order > 1:
            robust_mpc_narx.states = x_init
            robust_mpc_narx.inputs = u_init
            robust_mpc_narx.set_initial_guess()

        # setting initial guess to mpc if order == 1
        else:
            robust_mpc_narx.states = x_init
            robust_mpc_narx.set_initial_guess()

        start_time = time.time()

        # main loop
        for _ in range(iter):
            # random input inside the input boundaries
            u0_surrogate = robust_mpc_narx.make_step(x0=x0_real)

            # simulation steps
            x0_real = real_simulator.make_step(u0=u0_surrogate)

            # calculating closed loop cost
            cl_cost = -x0_real[1,0] + (u0_surrogate-u_prev).T @ r_mpc @ (u0_surrogate-u_prev)
            u_prev = u0_surrogate

            #closed_loop_cost.append(cqr_mpc.solver_stats['iterations']['obj'][-1])
            closed_loop_cost.append(cl_cost)
            opt_success.append(robust_mpc_narx.mpc.solver_stats['success'])

        end_time = time.time()

        violation_flag = self._check_boundary__violation(system=system, simulator=real_simulator)

        # storage
        self.simulation['simulator'] = real_simulator
        self.simulation['cs3'] = real_simulator

        # flag update
        self.flags.update({
            'simulation_ready': True,
        })

        # pass through
        self.cs3_results['Optimizer Success'] = all(opt_success)
        self.cs3_results['Closed Loop Cost'] = sum(closed_loop_cost) / len(closed_loop_cost)
        self.cs3_results['Boundary Violation'] = violation_flag
        self.cs3_results['Iterations'] = iter
        self.cs3_results['Time'] = end_time-start_time
        self.cs3_results['Initial Condition'] = ic_flag

        # end
        return self.cs3_results


    def setup_case_study_4(self, system, n_horizon, r, setpoint):

        # init
        real_model = system._get_model()
        real_mpc = system._get_mpc(model=real_model, n_horizon=n_horizon, r=r, setpoint=setpoint)

        self.cs4_mpc = MPC_NARX(mpc=real_mpc, n_x=self.data['n_x'], n_u=self.data['n_u'],
                                        order=self.data['order'])

        # preformance
        self.cs4_results = {'Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'Type': 'cs4',
                            'MPC Horizon': n_horizon,
                            'MPC Input Cost': r,
                            'MPC Setpoint': setpoint}

        return None


    def case_study_4(self, system, iter, x_init=None, u_init=None):

        # init
        order = self.data['order']
        cs4_mpc = self.cs4_mpc
        closed_loop_cost = []
        opt_success = []
        ic_flag = False if x_init is None else True

        r_mpc = self.cs4_results['MPC Input Cost'] * np.array([[0.1, 0],
                                                           [0,   1e-3]])
        u_prev = np.array([[0], [0]])

        # real model and simulator
        real_model = system._get_model()
        real_simulator = system._get_simulator(model=real_model)

        # random initial point is extracted if initial state not provided
        if x_init is None:
            # generate a new ic
            states_history, inputs = self._simulate_initial_guess(system=system, zero_ic=False, max_iter=10000)
            inputs_history = inputs[0:-1, :]

            # saving the new initial conditions
            x_init = states_history
            u_init = inputs_history

        # initial cond
        x0_real = x_init[0, :].reshape((-1, 1))
        real_simulator.x0 = x0_real
        real_simulator.set_initial_guess()

        # setting initial guess to mpc if order > 1
        if order > 1:
            cs4_mpc.states = x_init
            cs4_mpc.inputs = u_init
            cs4_mpc.set_initial_guess()

        # setting initial guess to mpc if order == 1
        else:
            cs4_mpc.states = x_init
            cs4_mpc.set_initial_guess()

        start_time = time.time()

        # main loop
        for _ in range(iter):
            # random input inside the input boundaries
            u0_surrogate = cs4_mpc.make_step(x0=x0_real)

            # simulation steps
            x0_real = real_simulator.make_step(u0=u0_surrogate)

            # calculating closed loop cost
            cl_cost = -x0_real[1,0] + (u0_surrogate-u_prev).T @ r_mpc @ (u0_surrogate-u_prev)
            u_prev = u0_surrogate

            #closed_loop_cost.append(cqr_mpc.solver_stats['iterations']['obj'][-1])
            closed_loop_cost.append(cl_cost)
            opt_success.append(cs4_mpc.mpc.solver_stats['success'])

        end_time = time.time()

        violation_flag = self._check_boundary__violation(system=system, simulator=real_simulator)

        # storage
        self.simulation['simulator'] = real_simulator
        self.simulation['cs4'] = real_simulator

        # flag update
        self.flags.update({
            'simulation_ready': True,
        })

        # pass through
        self.cs4_results['Optimizer Success'] = all(opt_success)
        self.cs4_results['Closed Loop Cost'] = sum(closed_loop_cost) / len(closed_loop_cost)
        self.cs4_results['Boundary Violation'] = violation_flag
        self.cs4_results['Iterations'] = iter
        self.cs4_results['Time'] = end_time-start_time
        self.cs4_results['Initial Condition'] = ic_flag



        # end
        return self.cs4_results

    def case_study_5(self, system, iter, n_horizon, r, tightner, rnd_samples,
                       confidence_cutoff, setpoint, max_search, store_gif=False, x_init=None, u_init=None):
        # init
        df = self.data['test']
        #narx_outputs = self.data['test_outputs']
        n_x = self.data['n_x']
        n_u = self.data['n_u']
        order = self.data['order']
        all_plots = []
        closed_loop_cost = []
        opt_success = []
        r_mpc = r * np.array([[0.1, 0],
                              [0,   1e-3]])
        u_prev = np.array([[0], [0]])
        ic_flag = False if x_init is None else True

        # system init
        model = system._get_model()
        simulator = system._get_simulator(model=model)
        estimator = do_mpc.estimator.StateFeedback(model= model)

        # additional config for the cqr
        self.cqr.set_config(rnd_samples=rnd_samples, confidence_cutoff=confidence_cutoff)

        # getting controller with surrogate model inside the mpc
        surrogate_model = self.narx_2_dompc()

        # creating the mpc class with the surrogate model
        surrogate_mpc = system._get_surrogate_mpc(surrogate_model=surrogate_model, n_x=n_x,
                                                  n_u=n_u, setpoint=setpoint, n_horizon=n_horizon, r=r)

        # storage
        cqr_mpc = MPC_Brancher_midterm(mpc=surrogate_mpc, cqr=self.cqr, confidence_cutoff=confidence_cutoff,
                                    tightner=tightner, max_search=max_search)

        # flag update
        self.flags.update({
            'cqr_mpc_ready': True,
        })

        # generate a new ic
        if x_init is None:
            states_history, inputs = self._simulate_initial_guess(system=system, zero_ic = False, max_iter = 10000)
            inputs_history = inputs[1:, :]

            x_init = states_history
            u_init = inputs_history

        # setting initial guess to mpc if order > 1
        if order > 1:
            cqr_mpc.states= x_init
            cqr_mpc.inputs= u_init
            cqr_mpc.set_initial_guess()

        # setting initial guess to mpc if order == 1
        else:
            cqr_mpc.states= x_init.reshape((-1, self.data['n_x']))
            cqr_mpc.set_initial_guess()

        # extracting the most recent initial state for the data
        x0 = x_init[0, :].reshape((n_x, 1))

        # setting initial guess to simulator
        simulator.x0=x0
        simulator.set_initial_guess()

        start_time = time.time()

        # run the main loop
        for i in range(iter):

            u0 = cqr_mpc.make_step(x0, enable_plots = store_gif)
            y0 = simulator.make_step(u0)
            x0 = estimator.make_step(y0)

            # calculating closed loop cost
            cl_cost = -x0[1,0] + (u0-u_prev).T @ r_mpc @ (u0-u_prev)
            u_prev = u0

            #closed_loop_cost.append(cqr_mpc.solver_stats['iterations']['obj'][-1])
            closed_loop_cost.append(cl_cost)
            opt_success.append(cqr_mpc.solver_stats['success'])

            print(f"\n\n++++#### Simulation report ####++++")
            print(f"Time: {simulator.t0}, Iteration: {i + 1} / {iter}")
            print(f"Input: {u0}")
            print(f"Measurement: {y0}")
            print(f"State Estimate: {x0}")
            print(f"++++#### End ####++++\n\n")

            if store_gif:
                all_plots.append(cqr_mpc.plot_trials_matplotlib(show_plot=False))

        end_time = time.time()

        violation_flag = self._check_boundary__violation(system=system, simulator=simulator)

        # storage
        self.simulation['simulator'] = simulator
        self.simulation['cs5'] = simulator
        self.store_gif = store_gif
        self.all_plots = all_plots

        # flag update
        self.flags.update({
                'simulation_ready': True,
            })

        # pass through
        self.cs5_results = {'Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'Type': 'cs5',
                            'No Samples': self.data['n_samples'],
                            'Surrogate Nodes': sum(self.narx.hidden_layers),
                            'CQR Nodes': sum(self.cqr.hidden_layers),
                            'CQR Alpha': self.cqr.alpha,
                            'Initial Condition': ic_flag,
                            'Optimizer Success': all(opt_success),
                            'Closed Loop Cost': sum(closed_loop_cost) / len(closed_loop_cost),
                            'Boundary Violation': violation_flag,
                            'Iterations': iter,
                            'MPC Horizon': n_horizon,
                            'MPC Input Cost': r,
                            'ICBMPC Tightner': tightner,
                            'ICBMPC Random Sampling':rnd_samples,
                            'ICBMPC Confidence Cutoff': confidence_cutoff,
                            'MPC Setpoint': setpoint,
                            'ICBMPC Max Search':max_search,
                            'Time': end_time-start_time}
        # end
        return self.cs5_results
    

    def combine_plots(self, system, x_limit_up = None, x_limit_down = None, y_limit_up = None, y_limit_down = None, figsize= None):
        """
        Combines all case study plots into a single figure.
        """

        assert self.flags['simulation_ready'] == True, \
            'Simulation data is not ready! Run the case studies first!'
        
        if figsize is None:
            figsize = (self.width_px, self.height_px)

        # init
        n_x = self.data['n_x']
        n_u = self.data['n_u']
        model = system._get_model()
        all_y_labels = model.x.keys() + model.u.keys()[1:] + model.aux.keys()[1:]
        simulators = [self.simulation['cs1'], self.simulation['cs2'], self.simulation['cs3'], self.simulation['cs4'], self.simulation['cs5']]
        labels = ['cs1', 'cs2', 'cs3', 'cs4', 'cs5']
        counter = 0

        fig, axes = plt.subplots(nrows=n_x + n_u, ncols=1, figsize=figsize, sharex=True)

        for i in range(n_x):
            ax = axes[i]

            # plot ii-th state simulation
            for ii, simulator in enumerate(simulators):
                ax.plot(simulator.data['_time'], simulator.data['_x'][:, i], label=labels[ii] if i ==0 else None)

            # plot system boundaries
            #ax.plot(simulator.data['_time'], [system.ubx[i]] * len(simulator.data['_time']), color='black', linestyle='solid', label='System Bounds' if i == 0 else None)
            #ax.plot(simulator.data['_time'], [system.lbx[i]] * len(simulator.data['_time']), color='black', linestyle='solid')

            # System bounds (upper and lower)
            upper_limit = np.full((simulator.data['_time'].shape[0],), system.ubx[i])
            lower_limit = np.full((simulator.data['_time'].shape[0],), system.lbx[i])

            # gray infill
            ax.fill_between(simulator.data['_time'].reshape(-1,), lower_limit, upper_limit, color='gray', alpha=0.5)

            # set y label
            ax.set_ylabel(all_y_labels[counter])
            counter += 1
            
        for j in range(n_u):
            ax = axes[n_x + j]

            # plot jj-th state simulation
            for jj, simulator in enumerate(simulators):
                ax.plot(simulator.data['_time'], simulator.data['_u'][:, j])

            # plot system boundaries
            #ax.plot(simulator.data['_time'], [system.ubu[j]] * len(simulator.data['_time']), color='black', linestyle='solid')
            #ax.plot(simulator.data['_time'], [system.lbu[j]] * len(simulator.data['_time']), color='black', linestyle='solid')

            # System bounds (upper and lower)
            upper_limit = np.full((simulator.data['_time'].shape[0],), system.ubu[j])
            lower_limit = np.full((simulator.data['_time'].shape[0],), system.lbu[j])

            # gray infill
            ax.fill_between(simulator.data['_time'].reshape(-1,), lower_limit, upper_limit, color='gray', alpha=0.5)

            # set y label
            ax.set_ylabel(all_y_labels[counter])
            counter += 1

        if y_limit_up is not None and y_limit_down is not None:
            for i, ax in enumerate(axes):
                ax.set_ylim(y_limit_down[i], y_limit_up[i])
            
        if x_limit_up is not None and x_limit_down is not None:
            for i, ax in enumerate(axes):
                ax.set_xlim(x_limit_down[i], x_limit_up[i])

        ax.set_xlabel('Time [s]')
        fig.legend(loc='upper right')
        fig.suptitle("Combined Case Studies")

        fig.show()

        return None
