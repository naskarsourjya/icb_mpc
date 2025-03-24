import numpy as np
import plotly.graph_objects as go


class MPC_Brancher():
    def __init__(self, mpc, cqr, tightner = 0.05, confidence_cutoff= 0.75):
        # storage
        #super(mpc_narx, self).__init__(model=model)     # `self` is the do_mpc MPC Controller
        self.mpc = mpc
        self.cqr = cqr
        self.tightner = tightner
        self.confidence_cutoff = confidence_cutoff

        # setup runtime
        self.setup()


    def setup(self):
        self._narxstates = None
        self._narxinputs = None
        self.t0 = 0.0
        self.history = None
        return None

    def reshape(self, array, shape):

        # rows and columns
        rows, cols = shape

        # end
        return array.reshape(cols, rows).T

    @property
    def states(self):
        return self._narxstates


    @states.setter
    def states(self, val):
        assert isinstance(val, np.ndarray), "states must be a numpy.array."

        assert val.shape[1] == self.cqr.order, \
            'Number of samples must be equal to the order of the NARX model!'

        assert val.shape[0] == self.cqr.n_x, (
            'Expected number of states is: {}, but found {}'.format(self.cqr.n_x, val.shape[0]))

        # storage
        self._narxstates = val


    @property
    def inputs(self):
        return self._narxinputs


    @inputs.setter
    def inputs(self, val):
        if self.cqr.order>1:
            assert isinstance(val, np.ndarray), "inputs must be a numpy.array."

            assert self.cqr.order - 1 == val.shape[1], \
                'Number of samples for inputs should be (order-1) !'

            assert val.shape[0] == self.cqr.n_u, (
                'Expected number of inputs is: {}, but found {}'.format(self.cqr.n_u, val.shape[0]))

            # storage
            self._narxinputs = val

        # error
        else:
            raise ValueError("Inputs cannot be set for system with order <= 1.")

    def _generate_initial_guess(self):

        # init
        states = self.states
        inputs = self.inputs

        order = self.cqr.order
        state_order = order
        input_order = order - 1

        state_samples = states.shape[1]
        input_samples = inputs.shape[1]

        # ensuring this is the current input
        # stacking states and inputs with order
        order_states = np.vstack([states[:, state_order - i - 1:state_samples - i] for i in range(state_order)])

        # if order is 2 or more, only then previous inputs are needed
        if order > 1:
            order_inputs = np.vstack([inputs[:, input_order - i - 1:input_samples - i] for i in range(input_order)])

            # stacking states and inputs for narx model
            initial_cond = np.vstack([order_states, order_inputs])

        else:
            initial_cond = order_states

        # storage
        self.initial_cond = initial_cond

        return initial_cond


    def set_initial_guess(self):

        # calculate initial guess
        initial_cond = self._generate_initial_guess()

        # passing initial cond
        self.mpc.x0 = initial_cond
        self.mpc.set_initial_guess()


        #self.mpc['history'] = None

        # end
        return None

    def make_step(self, x0, max_iter=10, enable_plots = False):

        assert x0.shape==(self.cqr.n_x, 1), \
            f"x0 should have shape ({self.cqr.n_x}, 1). Shape found instead is: {x0.shape}"

        # init
        initial_cond = self.initial_cond
        n_x = self.cqr.n_x
        n_u = self.cqr.n_u
        order = self.cqr.order
        # segregating states and inputs
        states = initial_cond[0:n_x * order, :]
        inputs = initial_cond[n_x * order:, :]
        narx_state_length = order * n_x + (order - 1) * n_u
        prev_ubx = self.mpc.bounds['upper', '_x', 'system_state']
        prev_lbx = self.mpc.bounds['lower', '_x', 'system_state']
        plots = []
        all_branches = []

        all_boundaries =[]

        # next pesudo step
        pseudo_input = self.inputs
        pseudo_state = np.hstack([self.states[:, 1:], x0])
        x0_stacked = np.vstack([x0, states[0:n_x * (order - 1)], inputs])

        # determining optimal input
        for i in range(max_iter):
            # init
            adjust_falg = False
            self.inputs = pseudo_input
            self.states = pseudo_state
            self.set_initial_guess()

            # making mpc prediction
            u0 = self.mpc.make_step(x0=x0_stacked)

            # extracting optimal trajectories
            u_traj = self.mpc.opt_x_num['_u']
            x_traj = self.mpc.opt_x_num['_x']

            # simulating cqr with this optimal trajectory
            x_traj_numpy = np.array([entry[0][0].full().flatten() for entry in x_traj]).T
            u_traj_numpy = np.array([entry[0][0].full().flatten() for entry in u_traj]).T

            x0_next = self.reshape(x_traj_numpy[0:n_x,0], shape=(n_x, 1))

            # setting up cqr
            self.cqr.states = pseudo_state
            self.cqr.inputs = pseudo_input
            self.cqr.set_initial_guess()

            # extraction of mpc boundaries
            ubx = self.mpc.bounds['upper', '_x', 'system_state']
            lbx = self.mpc.bounds['lower', '_x', 'system_state']
            boundaries = {}
            boundaries['lbx'] = lbx.full()[0: self.cqr.n_x, :]
            boundaries['ubx'] = ubx.full()[0: self.cqr.n_x, :]
            all_boundaries.append(boundaries)

            # make branch prediction
            branches = self.cqr.make_branch(u0_traj = u_traj_numpy, confidence_cutoff=self.confidence_cutoff)

            # reading the results
            all_states = np.hstack(branches['states'])

            # storage
            if enable_plots:
                all_branches.append(branches)
                plots.append(self.cqr.plot_branch(t0=self.t0, show_plot=False))

            # checking if all the predicted states saty inside the boundary
            for i in range(all_states.shape[1]):
                current_state = self.reshape(all_states[:,i], shape=(n_x, 1))
                if np.any(current_state < lbx.full()[0: self.cqr.n_x, :]) or np.all(current_state > ubx.full()[0: self.cqr.n_x, :]):
                    adjust_falg = True
                    break

            # exit loop
            if adjust_falg==False or i==max_iter-1:
                break

            # tighten up state boundaries
            else:
                range_x = ubx-lbx
                range_x = range_x[0:self.cqr.n_x, :]

                new_lbx = np.vstack([lbx[0:self.cqr.n_x, :] + self.tightner * range_x, np.full((narx_state_length - n_x, 1), -np.inf)])
                new_ubx = np.vstack([ubx[0:self.cqr.n_x, :] - self.tightner * range_x, np.full((narx_state_length - n_x, 1), np.inf)])

                self.mpc.bounds['upper', '_x', 'system_state'] = new_ubx
                self.mpc.bounds['lower', '_x', 'system_state'] = new_lbx

                self.mpc.reset_history()

        # pushing back the old boundaries for next make step
        self.mpc.bounds['upper', '_x', 'system_state'] = prev_ubx
        self.mpc.bounds['lower', '_x', 'system_state'] = prev_lbx

        # shifitng initial condition
        next_state_history = np.vstack([states[n_x:n_x * (order)], x0])
        next_input_history = np.vstack([inputs[n_u:n_u * (order - 1)], u0])

        # pushing it to class
        self.states = self.reshape(next_state_history, shape=(n_x, -1))
        self.inputs = self.reshape(next_input_history, shape=(n_u, -1))
        self._generate_initial_guess()

        # storage
        self.t0 += self.mpc.settings.t_step
        self.enable_plots = enable_plots

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
            history['time'].append(history['time'][-1] + self.mpc.settings.t_step)
            history['u0'] = np.hstack([history['u0'], u0])

            self.history = history


        if enable_plots:
            self.all_branches = all_branches
            self.plots = plots
            self.all_boundaries=all_boundaries

        # end
        return u0


    def plot_trials(self, show_plot=True):

        assert self.enable_plots, 'Plots storage not enabled! Set enable_plots=True in MPC_Brancher.make_step.'

        # init
        n_x = self.cqr.n_x
        n_u = self.cqr.n_u
        plots = self.plots
        history = self.history
        all_boundaries = self.all_boundaries
        new_plots = []

        for k, fig in enumerate(plots):
            branches = self.all_branches[k]
            boundaries = all_boundaries[k]
            lbx = boundaries['lbx'].reshape(-1,)
            ubx = boundaries['ubx'].reshape(-1,)
            branch_times = history['time'] + [num + history['time'][-1] for num in branches['time_stamps']]

            # add the past states
            for i in range(n_x):
                # making the mean prediction
                fig.add_trace(go.Scatter(x=history['time'], y=history['x0'][i,:],
                                         mode='lines',
                                         line=dict(color='black', dash='solid'), name='Simulation',
                                         showlegend=True if i==0 else False),
                              row=i + 1, col=1)

                # Add lines for system upper and lower bounds
                fig.add_trace(go.Scatter(x=branch_times,
                                         y=[self.cqr.ubx[i]] * len(branch_times), mode='lines',
                                         line=dict(color='grey', dash='solid'), name='System Bounds',
                                         showlegend=True if i == 0 else False),
                              row=i + 1, col=1)
                fig.add_trace(go.Scatter(x=branch_times,
                                         y=[self.cqr.lbx[i]] * len(branch_times), mode='lines',
                                         line=dict(color='grey', dash='solid'), name='System Bounds',
                                         showlegend=False),
                              row=i + 1, col=1)

                # Add lines for optimised upper and lower bounds
                fig.add_trace(go.Scatter(x=branch_times,
                                         y=[ubx[i]] * len(branch_times), mode='lines',
                                         line=dict(color='orange', dash='dash'), name='MPC Upper Bound',
                                         showlegend=True if i == 0 else False),
                              row=i + 1, col=1)
                fig.add_trace(go.Scatter(x=branch_times,
                                         y=[lbx[i]] * len(branch_times), mode='lines',
                                         line=dict(color='brown', dash='dash'), name='MPC Lower Bound',
                                         showlegend=True if i == 0 else False),
                              row=i + 1, col=1)

            for j in range(n_u):
                # making the mean prediction
                fig.add_trace(go.Scatter(x=history['time'], y=np.hstack([history['u0'][j,:-1], branches['u0_traj'][j,0]]),
                                         mode='lines',
                                         line=dict(color='black', dash='solid'),
                                         showlegend=False),
                              row=j + i + 2, col=1)

            if show_plot:
                fig.show()
            else:
                new_plots.append(fig)

        # end
        if show_plot:
            return None
        else:
            return new_plots

