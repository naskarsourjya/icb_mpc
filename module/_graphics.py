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
        self.height_px = 700
        self.width_px = 1800

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


    def plot_simulation(self):
        assert self.flags['simulation_ready'], 'Simulation not run! Run simulation first.'

        # using do-mpc for the plot
        fig, ax, graphics = do_mpc.graphics.default_plot(self.simulation['simulator'].data, figsize=(16, 9))
        graphics.plot_results()
        graphics.reset_axes()
        for ax_n in ax:
            ax_n.grid(True)
        plt.show()

        # end
        return None