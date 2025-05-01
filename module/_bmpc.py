import copy
import dill
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt


class MPC_Brancher():
    def __init__(self, mpc, cqr, tightner, confidence_cutoff, max_search, verbose=True):

        # sanity checks
        assert tightner > 0, "Tightner must be greater than 0."
        assert confidence_cutoff > 0 and confidence_cutoff <=1, "Confidence cutoff must be between 0 and 1."
        assert max_search >= 1 and isinstance(max_search, int), "Max search must be an integer greater than or equal to 1."
        
        # storage
        #super(mpc_narx, self).__init__(model=model)     # `self` is the do_mpc MPC Controller
        self.mpc = mpc
        self.cqr = cqr
        self.tightner = tightner
        self.confidence_cutoff = confidence_cutoff
        self.max_search = max_search
        self.verbose = verbose

        # setup runtime
        self.setup()


    def setup(self):
        self._narxstates = None
        self._narxinputs = None
        self.history = None
        return None


    @property
    def states(self):
        return self._narxstates


    @states.setter
    def states(self, val):
        assert isinstance(val, np.ndarray), "states must be a numpy.array."

        assert val.shape[0] == self.cqr.order, \
            'Number of samples must be equal to the order of the NARX model!'

        assert val.shape[1] == self.cqr.n_x, (
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

            assert self.cqr.order - 1 == val.shape[0], \
                'Number of samples for inputs should be (order-1) !'

            assert val.shape[1] == self.cqr.n_u, (
                'Expected number of inputs is: {}, but found {}'.format(self.cqr.n_u, val.shape[0]))

            # storage
            self._narxinputs = val

        # error
        else:
            raise ValueError("Inputs cannot be set for system with order <= 1.")

    def _generate_initial_guess(self, states, inputs=None):

        assert isinstance(states, np.ndarray), "states must be a numpy.array."

        assert states.shape[0] == self.cqr.order, \
            'Number of samples must be equal to the order of the NARX model!'

        assert states.shape[1] == self.cqr.n_x, (
            'Expected number of states is: {}, but found {}'.format(self.cqr.n_x, states.shape[0]))

        if self.cqr.order>1:

            assert isinstance(inputs, np.ndarray), "inputs must be a numpy.array."

            assert self.cqr.order - 1 == inputs.shape[0], \
                'Number of samples for inputs should be (order-1) !'

            assert inputs.shape[1] == self.cqr.n_u, (
                'Expected number of inputs is: {}, but found {}'.format(self.cqr.n_u, inputs.shape[0]))

            init_state = states.reshape((-1, 1))
            init_input = inputs.reshape((-1, 1))
            initial_cond = np.vstack([init_state,
                                      init_input])

        else:
            initial_cond = states.reshape((-1, 1))

        # end
        return initial_cond


    def set_initial_guess(self):

        # calculate initial guess
        initial_cond = self._generate_initial_guess(states=self.states, inputs=self.inputs)

        # resetting history
        self.history = None
        self.t0 = 0.0
        self.frame_number = 1

        # end
        return None


    def bounds_extractor(self, mpc, bnd_type, var_type):
        assert bnd_type == 'upper' or bnd_type == 'lower', "Only supported types are upper and lower."
        assert var_type == '_x' or var_type == '_u', "Only supported types are _x and _u."

        if var_type == '_x':
            for i in range(mpc.model.x.master.shape[0]):
                bnd_n = np.array(mpc.bounds[bnd_type, var_type, str(mpc.model.x.master[i, 0])])

                if i == 0:
                    bnd = bnd_n
                else:
                    bnd = np.vstack([bnd, bnd_n])

        elif var_type == '_u':
            for i in range(mpc.model.u.master.shape[0]):
                bnd_n = np.array(mpc.bounds[bnd_type, var_type, str(mpc.model.u.master[i, 0])])

                if i == 0:
                    bnd = bnd_n
                else:
                    bnd = np.vstack([bnd, bnd_n])

        return bnd


    def bounds_setter(self, mpc, bnd_type, var_type, bnd_val):
        assert bnd_type == 'upper' or bnd_type == 'lower', "Only supported types are upper and lower."
        assert var_type == '_x' or var_type == '_u', "Only supported types are _x and _u."

        if var_type == '_x':
            for i, bnd_val_n in enumerate(bnd_val.flatten().tolist()):
                mpc.bounds[bnd_type, var_type, str(mpc.model.x.master[i, 0])] = bnd_val_n

        elif var_type == '_u':
            for i, bnd_val_n in enumerate(bnd_val.flatten().tolist()):
                mpc.bounds[bnd_type, var_type, str(mpc.model.u.master[i, 0])] = bnd_val_n


        return None



    def make_step(self, x0, enable_plots = False):

        assert x0.shape==(self.cqr.n_x, 1), \
            f"x0 should have shape ({self.cqr.n_x}, 1). Shape found instead is: {x0.shape}"

        # init
        all_branches = []
        plots = []
        all_boundaries = []
        prev_state = self.states
        prev_input = self.inputs

        # take the new x0 and generate new initial condition
        current_state = np.vstack([x0.reshape((1, -1)),
                                  self.states[:-1, :]])
        x0_current = self._generate_initial_guess(states=current_state, inputs=prev_input)

        # extracting and storing boundaries
        default_lbx = self.bounds_extractor(mpc=self.mpc, bnd_type='lower', var_type='_x')
        default_ubx  = self.bounds_extractor(mpc=self.mpc, bnd_type='upper', var_type='_x')

        for i in range(self.max_search):

            # debug printouts
            if self.verbose:
                print(f"\n\n--------- Iterative Boundary Nominal Model Predictive Control ---------")
                print(f"Frame Number: {self.frame_number}")
                print(f"Time: {self.t0}, Iteration: {i + 1} / {self.max_search}\n\n")

            # re-init to old state
            prev_ic = self._generate_initial_guess(states=prev_state, inputs=prev_input)

            # passing initial cond
            self.mpc.x0 = prev_ic
            self.mpc.set_initial_guess()

            # do make step
            u0 = self.mpc.make_step(x0=x0_current)

            lbx = self.bounds_extractor(mpc=self.mpc, bnd_type='lower', var_type='_x')
            ubx = self.bounds_extractor(mpc=self.mpc, bnd_type='upper', var_type='_x')
            boundaries = {}
            boundaries['lbx'] = lbx
            boundaries['ubx'] = ubx
            all_boundaries.append(boundaries)

            # extracting optimal trajectories
            u_traj = self.mpc.opt_x_num['_u']
            u_traj_numpy = np.array([entry[0][0].full().flatten() for entry in u_traj])

            # setting up cqr
            self.cqr.states = current_state
            if self.cqr.order>1:
                self.cqr.inputs = self.inputs
            self.cqr.set_initial_guess()

            # make branch prediction
            branches = self.cqr.make_branch(u0_traj=u_traj_numpy)

            # storage
            if enable_plots:
                all_branches.append(branches)
                plots.append(self.cqr.plot_branch_matplotlib(t0=self.t0, show_plot=False))

            # adjust bounds if necessary
            adjust_flag = self._adjust_bounds(mpc=self.mpc, branches=branches,
                                              default_lbx = default_lbx, default_ubx=default_ubx)
            
            # debug printouts
            if self.verbose:
                print(f"\n\nBoundary adjusted: {adjust_flag}")
                print(f"State Upper Bound: {self.bounds_extractor(mpc=self.mpc, bnd_type='upper', var_type='_x')}")
                print(f"State Lower Bound: {self.bounds_extractor(mpc=self.mpc, bnd_type='lower', var_type='_x')}")
                print(f"Input Upper Bound: {self.bounds_extractor(mpc=self.mpc, bnd_type='upper', var_type='_u')}")
                print(f"Input Lower Bound: {self.bounds_extractor(mpc=self.mpc, bnd_type='lower', var_type='_u')}")
                print(f"Time: {self.t0}, Iteration: {i + 1} / {self.max_search}")
                print(f"Calculated input: {u0}")
                print(f"Calculated next state: {branches['states'][1][0,:]}")
                print(f"--------- End ---------\n\n")
            
            # increment frame number
            self.frame_number += 1

            if adjust_flag == False:
                break

        # revert back to the system boundaries for the state
        self.bounds_setter(mpc=self.mpc, bnd_type='upper', var_type='_x', bnd_val=default_ubx)
        self.bounds_setter(mpc=self.mpc, bnd_type='lower', var_type='_x', bnd_val=default_lbx)

        # push the new u0 into the new initial condition
        if self.cqr.order>1:
            pseudo_input_history = np.vstack([u0.reshape((1, -1)),
                                              self.inputs[:-1, :]])
            self.inputs = pseudo_input_history
        self.states = current_state

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
            self.all_boundaries = all_boundaries

        # end
        return u0

    def _adjust_bounds(self, mpc, branches, default_lbx, default_ubx):
        # init
        adjust_flag = False

        # checking if all the predicted states saty inside the boundary
        # for i in range(all_states.shape[0]):
        #    current_state = all_states[i, :].reshape((1, -1))
        #    if (np.any(current_state < lbx.reshape((-1,))[0: self.cqr.n_x]) or
        #            np.any(current_state > ubx.reshape((-1,))[0: self.cqr.n_x])):
        #        adjust_flag = True
        #        break

        all_states = np.vstack(branches['states'])

        default_lbx_matrix = np.vstack([default_lbx.reshape((-1,))[0: self.cqr.n_x]] * all_states.shape[0])
        default_ubx_matrix = np.vstack([default_ubx.reshape((-1,))[0: self.cqr.n_x]] * all_states.shape[0])

        if (np.any(all_states < default_lbx_matrix) or np.any(all_states > default_ubx_matrix)):
            adjust_flag = True

        # exit loop
        if adjust_flag == True:

            # calculation per alpha value
            for i, states_n in enumerate(branches['states']):

                # init
                default_lbx_matrix_n = np.vstack([default_lbx.reshape((-1,))[0: self.cqr.n_x]] * states_n.shape[0])
                default_ubx_matrix_n = np.vstack([default_ubx.reshape((-1,))[0: self.cqr.n_x]] * states_n.shape[0])
                default_range_x_n = default_ubx_matrix_n - default_lbx_matrix_n
                alpha_n = branches['alphas'][i]

                # checking the protrusions above the upper boundary, if positive: boundary is crossed
                residue_upper = (states_n - default_ubx_matrix_n) / default_range_x_n

                # checking the protrusions below the lower boundary, if positive: boundary is crossed
                residue_lower = (- states_n + default_lbx_matrix_n) / default_range_x_n

                # Replace values: 1 for zero or positive, 0 for negative
                # if value is 1, then the boundary is violated, if value is 0, boundary not violated.
                binary_upper = np.where(residue_upper >= 0, 1, 0)
                binary_lower = np.where(residue_lower >= 0, 1, 0)

                # replacing all states with zero, which have not crossed the boundary
                residue_upper[residue_upper < 0] = 0
                residue_lower[residue_lower < 0] = 0

                # if boundary is  too far, i.e., the difference itself is more than the range of x,
                # it is set at the range of the x
                residue_upper[residue_upper > 1] = 1
                residue_lower[residue_lower > 1] = 1

                # calculating the probability of the states crossing the individual boundary
                residue_prob_u = np.sum(binary_upper, axis=0) / states_n.shape[0]
                residue_prob_l = np.sum(binary_lower, axis=0) / states_n.shape[0]

                # probabilities scaled with the cqr alpha value
                upper_prob = alpha_n * residue_prob_u
                lower_prob = alpha_n * residue_prob_l
                upper_prob_stacked = np.vstack([upper_prob] * states_n.shape[0])
                lower_prob_stacked = np.vstack([lower_prob] * states_n.shape[0])

                if i == 0:
                    prob_scaled_states_u = upper_prob_stacked * residue_upper
                    prob_scaled_states_l = lower_prob_stacked * residue_lower

                else:
                    prob_scaled_states_u = np.vstack([prob_scaled_states_u, upper_prob_stacked * residue_upper])
                    prob_scaled_states_l = np.vstack([prob_scaled_states_l, lower_prob_stacked * residue_lower])

            # determining the mean of the deviations
            prob_upper = np.mean(prob_scaled_states_u, axis=0)
            prob_lower = np.mean(prob_scaled_states_l, axis=0)

            # extracting current mpc bounds
            lbx = self.bounds_extractor(mpc=mpc, bnd_type='lower', var_type='_x')
            ubx = self.bounds_extractor(mpc=mpc, bnd_type='upper', var_type='_x')

            range_x = ubx - lbx
            range_x = range_x[0:self.cqr.n_x, :]

            # adjusting the bounds
            adj_lower = self.tightner * prob_lower * range_x.reshape((-1,))
            adj_upper = self.tightner * prob_upper * range_x.reshape((-1,))

            new_lbx = lbx[0:self.cqr.n_x, :] + adj_lower.reshape((-1, 1))
            new_ubx = ubx[0:self.cqr.n_x, :] - adj_upper.reshape((-1, 1))

            # sanity check
            assert np.all(new_lbx < new_ubx), "Lower bound has to be is greater than upper bound! Reduce 'tightner' to ensure this does not occur."

            # readjusting boundaries
            self.bounds_setter(mpc=mpc, bnd_type='upper', var_type='_x', bnd_val=new_ubx)
            self.bounds_setter(mpc=mpc, bnd_type='lower', var_type='_x', bnd_val=new_lbx)

            # resetting history
            self.mpc.reset_history()

        return adjust_flag


    def _adjust_bounds_v(self, mpc, branches, default_lbx, default_ubx):
        # init
        adjust_flag = False

        # checking if all the predicted states saty inside the boundary
        #for i in range(all_states.shape[0]):
        #    current_state = all_states[i, :].reshape((1, -1))
        #    if (np.any(current_state < lbx.reshape((-1,))[0: self.cqr.n_x]) or
        #            np.any(current_state > ubx.reshape((-1,))[0: self.cqr.n_x])):
        #        adjust_flag = True
        #        break

        all_states = np.vstack(branches['states'])

        default_lbx_matrix = np.vstack([default_lbx.reshape((-1,))[0: self.cqr.n_x]]*all_states.shape[0])
        default_ubx_matrix = np.vstack([default_ubx.reshape((-1,))[0: self.cqr.n_x]] * all_states.shape[0])

        if (np.any(all_states < default_lbx_matrix) or np.any(all_states > default_ubx_matrix)):
            adjust_flag = True

        # exit loop
        if adjust_flag == True:

            # init
            alpha_list = []

            # calculation per alpha value
            for i, states_nth in enumerate(branches['states']):
                alpha_nth = [branches['alphas'][i]] * states_nth.shape[0]

                alpha_list = alpha_list + alpha_nth

            # stacked alpha
            alpha_stacked = np.stack(alpha_list)

            # init
            default_range_x = default_ubx_matrix - default_lbx_matrix

            # checking the protrusions above the upper boundary, if positive: boundary is crossed
            residue_upper = (all_states - default_ubx_matrix) / default_range_x

            # checking the protrusions below the lower boundary, if positive: boundary is crossed
            residue_lower = (- all_states + default_lbx_matrix) / default_range_x

            # Replace values: 1 for zero or positive, 0 for negative
            # if value is 1, then the boundary is violated, if value is 0, boundary not violated.
            binary_upper = np.where(residue_upper >= 0, 1, 0)
            binary_lower = np.where(residue_lower >= 0, 1, 0)

            # replacing all states with zero, which have not crossed the boundary
            residue_upper[residue_upper < 0] = 0
            residue_lower[residue_lower < 0] = 0

            # if boundary is  too far, i.e., the difference itself is more than the range of x,
            # it is set at the range of the x
            residue_upper[residue_upper > 1] = 1
            residue_lower[residue_lower > 1] = 1

            # calculating the probability of the states crossing the individual boundary
            residue_prob_u = np.sum(binary_upper, axis=0) / all_states.shape[0]
            residue_prob_l = np.sum(binary_lower, axis=0) / all_states.shape[0]

            # probabilities scaled with the cqr alpha value
            upper_prob = alpha_stacked * residue_prob_u
            lower_prob = alpha_stacked * residue_prob_l
            upper_prob_stacked = np.vstack([upper_prob] * all_states.shape[0])
            lower_prob_stacked = np.vstack([lower_prob] * all_states.shape[0])

            # all scaled deviations
            prob_scaled_states_u = upper_prob_stacked * residue_upper
            prob_scaled_states_l = lower_prob_stacked * residue_lower

            # determining the mean of the deviations
            prob_upper = np.mean(prob_scaled_states_u, axis=0)
            prob_lower = np.mean(prob_scaled_states_l, axis=0)

            # extracting current mpc bounds
            lbx = self.bounds_extractor(mpc=mpc, bnd_type='lower', var_type='_x')
            ubx = self.bounds_extractor(mpc=mpc, bnd_type='upper', var_type='_x')

            range_x = ubx - lbx
            range_x = range_x[0:self.cqr.n_x, :]

            # adjusting the bounds
            adj_lower = self.tightner * prob_lower * range_x.reshape((-1,))
            adj_upper = self.tightner * prob_upper * range_x.reshape((-1,))

            new_lbx = lbx[0:self.cqr.n_x, :] + adj_lower.reshape((-1, 1))
            new_ubx = ubx[0:self.cqr.n_x, :] - adj_upper.reshape((-1, 1))

            assert np.all(new_lbx < new_ubx), "Lower bound has to be is greater than upper bound! Reduce self.tightner to ensure this does not occur."

            # self.mpc.bounds['upper', '_x', 'system_state'] = new_ubx
            # self.mpc.bounds['lower', '_x', 'system_state'] = new_lbx
            self.bounds_setter(mpc=mpc, bnd_type='upper', var_type='_x', bnd_val=new_ubx)
            self.bounds_setter(mpc=mpc, bnd_type='lower', var_type='_x', bnd_val=new_lbx)

            self.mpc.reset_history()

        return adjust_flag



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


    def plot_trials_matplotlib(self, show_plot=True):
        assert self.enable_plots, 'Plots storage not enabled! Set enable_plots=True in MPC_Brancher.make_step.'

        # Initialize parameters
        n_x = self.cqr.n_x
        n_u = self.cqr.n_u
        plots = self.plots  # Assuming this is a list of figures (not used directly here)
        history = self.history
        all_boundaries = self.all_boundaries

        new_plots = []

        for k in range(len(plots)):
            branches = self.all_branches[k]
            boundaries = all_boundaries[k]
            lbx = boundaries['lbx'].reshape(-1, )
            ubx = boundaries['ubx'].reshape(-1, )

            branch_times = history['time'] + [num + history['time'][-1] for num in branches['time_stamps']]

            fig, axes = plots[k]

            # Loop through each state variable (n_x)
            for i in range(n_x):
                ax = axes[i]

                # Simulation line (history of states)
                ax.plot(history['time'], history['x0'][i, :], color='black', linestyle='solid',
                        label='Simulation' if i == 0 else None)

                # System bounds (upper and lower)
                ax.plot(branch_times, [self.cqr.ubx[i]] * len(branch_times), color='grey', linestyle='solid',
                        label='System Bounds' if i == 0 else None)
                ax.plot(branch_times, [self.cqr.lbx[i]] * len(branch_times), color='grey', linestyle='solid')

                # Optimized MPC bounds (upper and lower)
                ax.plot(branch_times, [ubx[i]] * len(branch_times), color='orange', linestyle='dashed',
                        label='MPC Upper Bound' if i == 0 else None)
                ax.plot(branch_times, [lbx[i]] * len(branch_times), color='brown', linestyle='dashed',
                        label='MPC Lower Bound' if i == 0 else None)

                #ax.set_ylabel(f'State {i + 1}')

            # Loop through each control variable (n_u)
            for j in range(n_u):
                ax = axes[n_x + j]

                # Combine historical control inputs with the first value from branches
                u_combined = np.hstack([history['u0'][j, :-1], branches['u0_traj'][j, 0]])

                ax.plot(history['time'], u_combined, color='black', linestyle='solid')

                #ax.set_ylabel(f'Input {j + 1}')

            # Set x-axis labels on all plots after looping through inputs and states.
            #for ax in axes:
            #    ax.set_xlabel('Time [s]')

            fig.suptitle("MPC Trial Plots", fontsize=16)
            fig.legend(loc='upper right')

            if show_plot:
                fig.show()
            else:
                plt.close(fig)
                new_plots.append((fig, axes))

        if show_plot:
            return None
        else:
            return new_plots
