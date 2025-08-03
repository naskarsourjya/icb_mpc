import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import do_mpc
import torch
from tqdm import tqdm

class plotter():
    def __init__(self):

        # plottting row size
        self.height_px = 6
        self.width_px = 10

    def visualize2d_data(self):

        assert self.flags['data_stored'] == True,\
            'Data not found! First run random_trajectory_sampler(), to generate data.'
        assert self.data['n_x'] == 2, "This function only for n_x = 2. Alternatively you can try visualize_data()."

        # init
        df = self.data['simulation']

        # setting up plot
        fig, ax = plt.subplots(1 + self.data['n_u'], figsize=(self.width_px, self.height_px))
        fig.suptitle('Input and State space plot')

        ax[0].plot(df['state_1'], df['state_2'],)

        # Define the limits
        x_lower, x_upper = self.data['lbx'][0], self.data['ubx'][0]  # Limits for the x-axis
        y_lower, y_upper = self.data['lbx'][1], self.data['ubx'][1]  # Limits for the y-axis

        # Plot the box with gray infill
        rect = plt.Rectangle((x_lower, y_lower), x_upper - x_lower, y_upper - y_lower,
                             color='gray', alpha=0.5)
        ax[0].add_patch(rect)

        ax[0].set_xlabel('state_1')
        ax[0].set_ylabel('state_2')

        for i in range(self.data['n_u']):
            col_name = f'input_{i+1}'

            # input plot
            ax[i+1].plot(df['time'], df[col_name])

            # generating upper and lower limit
            upper_limit = np.full((df.shape[0],), self.data['ubu'][i])
            lower_limit = np.full((df.shape[0],), self.data['lbu'][i])

            # Plot upper and lower limits
            ax[i+1].plot(df['time'], upper_limit, linestyle='dashed', color='green')
            ax[i+1].plot(df['time'], lower_limit, linestyle='dashed', color='red')

            # gray infill
            ax[i+1].fill_between(df['time'], lower_limit, upper_limit, color='gray', alpha=0.5)
            label = 'input_' + str(i + 1)
            ax[i+1].set_ylabel(label)

        ax[-1].set_xlabel('time')
        fig.legend()
        plt.show()

        return None


    def visualize_data(self):
        assert self.flags['data_stored'] == True, \
            'Data not found! First run random_trajectory_sampler(), to generate data.'

        # init
        df = self.data['simulation']

        # setting up plot
        fig, ax = plt.subplots(self.data['n_x'] + self.data['n_u'],
                               figsize=(self.width_px, self.height_px))
        fig.suptitle('Input and State space plot')

        for i in range(self.data['n_x']):

            # init
            col_name = f'state_{i + 1}'

            # plot state
            ax[i].plot(df['time'], df[col_name])

            # generating upper and lower limit
            upper_limit = np.full((df.shape[0],), self.data['ubx'][i])
            lower_limit = np.full((df.shape[0],), self.data['lbx'][i])

            # Plot upper and lower limits
            ax[i].plot(df['time'], upper_limit, linestyle='dashed', color='green')
            ax[i].plot(df['time'], lower_limit, linestyle='dashed', color='red')

            # gray infill
            ax[i].fill_between(df['time'], lower_limit, upper_limit, color='gray', alpha=0.5)

            ax[i].set_ylabel(col_name)

        for i in range(self.data['n_u']):
            # init
            col_name = f'input_{i + 1}'

            # plot input
            ax[i+self.data['n_x']].plot(df['time'], df[col_name])

            # generating upper and lower limit
            upper_limit = np.full((df.shape[0],), self.data['ubu'][i])
            lower_limit = np.full((df.shape[0],), self.data['lbu'][i])

            # Plot upper and lower limits
            ax[i+self.data['n_x']].plot(df['time'], upper_limit, linestyle='dashed', color='green')
            ax[i+self.data['n_x']].plot(df['time'], lower_limit, linestyle='dashed', color='red')

            # gray infill
            ax[i+self.data['n_x']].fill_between(df['time'], lower_limit, upper_limit, color='gray', alpha=0.5)


            ax[i + self.data['n_x']].set_ylabel(col_name)

        ax[-1].set_xlabel('time')
        fig.legend()
        plt.show()

        return None


    def plot_simulation(self, system, simulator= None, x_limit_up = None, x_limit_down = None, y_limit_up = None,
                        y_limit_down = None, figsize= None, fig_name=None):

        # using do-mpc for the plot
        n_x = self.data['n_x']
        n_u = self.data['n_u']
        all_y_labels = ['$C_a [mol/L]$', '$C_b [mol/L]$', '$T_R [{}^\circ C]$', '$T_K [{}^\circ C]$', '$F [L/h]$', '$Q_{dot} [kJ/h]$']

        if figsize is None:
            figsize = (self.width_px, self.height_px)

        if simulator is None:
            assert self.flags['simulation_ready'], 'Simulation not run! Run simulation first.'
            simulator = self.simulation['simulator']

        #fig, axes, graphics = do_mpc.graphics.default_plot(simulator.data, figsize=figsize)
        #graphics.plot_results()
        #graphics.reset_axes()

        fig, axes = plt.subplots(nrows=n_x + n_u, ncols=1, figsize=figsize, sharex=True)


        for i, ax_n in enumerate(axes):
            if i < n_x:

                # plot simulation
                ax_n.plot(simulator.data['_time'].reshape(-1,), simulator.data['_x'][:, i].reshape(-1,), label= 'trajectory' if i == 0 else None)

                # System bounds (upper and lower)
                upper_limit = np.full((simulator.data['_time'].shape[0],), system.ubx[i])
                lower_limit = np.full((simulator.data['_time'].shape[0],), system.lbx[i])

                # gray infill
                ax_n.fill_between(simulator.data['_time'].reshape(-1,), lower_limit, upper_limit, color='lightgrey',
                                  alpha=0.5, label='system bounds' if i == 0 else None)
                #ax_n.plot(simulator.data['_time'].reshape(-1,), upper_limit, linestyle='dashed', color='black', label='upper limit' if i == 0 else None)
                #ax_n.plot(simulator.data['_time'].reshape(-1,), lower_limit, linestyle='dashed', color='black')
                
                ax_n.set_ylabel(all_y_labels[i])



            elif i>=n_x and i<(n_x+n_u):

                # plot simulation
                ax_n.plot(simulator.data['_time'].reshape(-1,), simulator.data['_u'][:, i-n_x].reshape(-1,))

                # System bounds (upper and lower)
                upper_limit = np.full((simulator.data['_time'].shape[0],), system.ubu[i-n_x])
                lower_limit = np.full((simulator.data['_time'].shape[0],), system.lbu[i-n_x])

                # gray infill
                ax_n.fill_between(simulator.data['_time'].reshape(-1,), lower_limit, upper_limit, color='lightgrey',
                                  alpha=0.5)
                #ax_n.plot(simulator.data['_time'].reshape(-1,), upper_limit, linestyle='dashed', color='black')
                #ax_n.plot(simulator.data['_time'].reshape(-1,), lower_limit, linestyle='dashed', color='black')
                
                ax_n.set_ylabel(all_y_labels[i])

        if y_limit_up is not None and y_limit_down is not None:
            for i, ax in enumerate(axes):
                ax.set_ylim(y_limit_down[i], y_limit_up[i])

        if x_limit_up is not None and x_limit_down is not None:
            for i, ax in enumerate(axes):
                ax.set_xlim(x_limit_down[i], x_limit_up[i])

        axes[-1].set_xlabel('$Time [h]$')
        fig.legend(loc='upper left', bbox_to_anchor=(1.00, 1))
        fig.show()

        # save figure as pdf
        if fig_name is not None:
            fig.savefig(fig_name, bbox_inches='tight', format='pdf')

        # end
        return None
