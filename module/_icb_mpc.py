import torch
import do_mpc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science','no-latex'])


class ICB_MPC():
    def __init__(self, mpc, cqr, tightner, confidence_cutoff, max_search, R, Q, verbose=True):

        # sanity checks
        assert tightner > 0, "Tightner must be greater than 0."
        assert confidence_cutoff > 0 and confidence_cutoff <= 1, "Confidence cutoff must be between 0 and 1."
        assert max_search >= 1 and isinstance(max_search,
                                              int), "Max search must be an integer greater than or equal to 1."

        # storage
        # super(mpc_narx, self).__init__(model=model)     # `self` is the do_mpc MPC Controller
        self.mpc = mpc
        self.cqr = cqr
        self.tightner = tightner
        self.confidence_cutoff = confidence_cutoff
        self.max_search = max_search
        self.verbose = verbose
        self.R = R
        self.Q = Q

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
        if self.cqr.order > 1:
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

        if self.cqr.order > 1:

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

    def make_step_nobranch(self, x0):

        assert x0.shape == (self.cqr.n_x, 1), \
            f"x0 should have shape ({self.cqr.n_x}, 1). Shape found instead is: {x0.shape}"

        prev_state = self.states
        prev_input = self.inputs

        # take the new x0 and generate new initial condition
        current_state = np.vstack([x0.reshape((1, -1)),
                                   self.states[:-1, :]])
        x0_current = self._generate_initial_guess(states=current_state, inputs=prev_input)
        prev_ic = self._generate_initial_guess(states=prev_state, inputs=prev_input)

        # passing initial cond
        self.mpc.x0 = prev_ic
        self.mpc.set_initial_guess()

        # do make step
        u0 = self.mpc.make_step(x0=x0_current)

        # push the new u0 into the new initial condition
        if self.cqr.order > 1:
            pseudo_input_history = np.vstack([u0.reshape((1, -1)),
                                              self.inputs[:-1, :]])
            self.inputs = pseudo_input_history
        self.states = current_state

        # end
        return u0

    def make_branch(self, u0_traj):
        assert self.cqr.flags['qr_ready'], "Quantile regressor not ready."
        assert self.cqr.flags['cqr_ready'], "Quantile regressor not conformalised."
        assert self.cqr.flags['cqr_initial_condition_ready'], "CQR not initialised"
        assert u0_traj.shape[1] == self.cqr.n_u, \
            f"u0 should have have {self.cqr.n_u} columns but instead found {u0_traj.shape[1]}!"

        # storage for later retrieval
        n_x = self.cqr.n_x
        n_u = self.cqr.n_u
        order = self.cqr.order
        state_n = self.cqr.states.reshape((1, -1))
        if self.cqr.order > 1:
            input_n = self.cqr.inputs.reshape((1, -1))
        alpha_branch = [1]
        time_branch = [0.0]
        steps = u0_traj.shape[0]
        states_branch = [self.cqr.states[0, :].reshape(1, -1)]
        inputs_branch = []
        input_labels = [col for col in self.cqr.narx.x_label if col.endswith('lag_0') and col.startswith('input')]

        # generating the branches
        for i in range(steps):

            # init
            u0 = u0_traj[i, :].reshape((1, -1))
            n_samples = state_n.shape[0]

            # segregating states and inputs
            u0_stacked = np.vstack([u0] * n_samples)

            # stacking all data
            if self.cqr.order > 1:
                X = np.hstack([state_n, u0_stacked, input_n])
            else:
                X = np.hstack([state_n, u0_stacked])

            # setting default device
            self.cqr._set_device(torch_device=self.cqr.full_model.torch_device)

            # narx_input = self.cqr.input_preprocessing(states=order_states, inputs=order_inputs)
            X = pd.DataFrame(data=X, columns=self.cqr.narx.x_label)

            # get robust input
            X_robust = self.get_robust_input(X=X)

            # converting to torch
            X_torch = torch.tensor(X_robust.to_numpy(), dtype=self.cqr.dtype)

            # making full model prediction
            with torch.no_grad():
                y_pred = self.cqr.full_model(X_torch)

            # doing postprocessing containing the conformalisation step
            x0, x0_cqr_high, x0_cqr_low = self.cqr._post_processing(y=y_pred)

            # finding the upper limit
            states_3d = np.stack([x0, x0_cqr_high, x0_cqr_low], axis=0)

            # finding the limits per row and col
            max_states = np.max(states_3d, axis=0)
            min_states = np.min(states_3d, axis=0)

            # sanity check
            assert np.all(max_states >= min_states), ("Some values of the in the max_states in < min_states, "
                                                      "which is not expected. Should not happen if "
                                                      "the system is monotonic.")

            # generating random points between the max and the min
            random_states = np.random.uniform(
                low=min_states,
                high=max_states,
                size=(self.cqr.rnd_samples, *x0.shape)
            )

            random_states_2d = random_states.reshape((-1, self.cqr.n_x))

            # stacking outputs
            x0_next = np.vstack([x0, x0_cqr_high, x0_cqr_low, random_states_2d])

            # preparing the next initial conditions
            state_n = np.hstack([x0_next, np.vstack([state_n] * (3 + self.cqr.rnd_samples))])[:, 0:n_x * order]
            if self.cqr.order > 1:
                input_n = np.hstack(
                    [np.vstack([u0] * x0_next.shape[0]), np.vstack([input_n] * (3 + self.cqr.rnd_samples))])[:,
                          0:n_u * (order - 1)]

            # stores the branched states
            if self.cqr.confidence_cutoff == 1:

                # branches not stored if confidence_cutoff = 1, equivalent to nominal mpc
                states_branch.append(np.vstack([x0]))

            else:
                states_branch.append(x0_next)

            inputs_branch.append(X_robust[input_labels].to_numpy())
            alpha_branch.append(alpha_branch[-1] * (1 - self.cqr.alpha))
            time_branch.append(time_branch[-1] + self.cqr.t_step)

            # force cutoff is confidence is low
            if alpha_branch[-1] < self.cqr.confidence_cutoff:
                break

        # storage
        self.cqr.branches = {'states': states_branch,
                             'alphas': alpha_branch,
                             'time_stamps': time_branch,
                             'u0_traj': u0_traj,
                             'inputs': inputs_branch}

        # end
        return self.cqr.branches

    def plot_branch_matplotlib(self, t0=0.0, show_plot=True):
        n_x = self.cqr.n_x
        n_u = self.cqr.n_u
        time_stamp_states = [num + t0 for num in self.cqr.branches['time_stamps']]
        states = self.cqr.branches['states']
        inputs = self.cqr.branches['inputs']
        alphas = self.cqr.branches['alphas']
        u0_traj = self.cqr.branches['u0_traj']
        time_stamp_inputs = np.arange(t0, t0 + (self.cqr.t_step * u0_traj.shape[0]), self.cqr.t_step)[
                            0:u0_traj.shape[0]]

        # Create subplots
        fig, axes = plt.subplots(n_x + n_u, 1, figsize=(self.cqr.width_px, self.cqr.height_px), sharex=True)

        if n_x + n_u == 1:  # If there's only one subplot, wrap axes in a list for consistency
            axes = [axes]

        # Loop through each state
        for i in range(n_x):
            ax = axes[i]
            mean_prediction = []

            for j, t in enumerate(time_stamp_states):
                if j < len(time_stamp_states) - 1:
                    # Add shaded region for confidence
                    ax.fill_between(
                        [time_stamp_states[j], time_stamp_states[j + 1]],
                        [min(states[j][:, i]), min(states[j + 1][:, i])],
                        [max(states[j][:, i]), max(states[j + 1][:, i])],
                        color='yellow',
                        alpha=alphas[j],
                        label=f'Confidence' if j == 0 and i == 0 else None,
                    )

                # Scatter plot of branches
                ax.scatter([t] * states[j][:, i].shape[0], states[j][:, i], color='pink', s=2,
                           label='Branches' if i == 0 and j == 0 else None)

                # Extracting mean prediction
                mean_prediction.append(states[j][0, i])

            # Line plot of mean prediction
            ax.plot(time_stamp_states, mean_prediction,
                    linestyle='dashed', color='red', label='Nominal Projection' if i == 0 else None)

            ax.set_ylabel(f'State {i + 1}')
            # ax.legend()

        for i in range(n_u):
            ax = axes[n_x + i]

            for j, t in enumerate(time_stamp_inputs[:len(inputs)]):

                if j < len(time_stamp_inputs[:len(inputs)]) - 1:
                    # Add shaded region for confidence
                    ax.fill_between(
                        [time_stamp_states[j], time_stamp_states[j + 1]],
                        [min(inputs[j][:, i]), min(inputs[j + 1][:, i])],
                        [max(inputs[j][:, i]), max(inputs[j + 1][:, i])],
                        color='yellow',
                        alpha=alphas[j],
                        label=f'Confidence' if j == 0 and i == 0 else None,
                    )

                # Scatter plot of branches
                ax.scatter([t] * inputs[j][:, i].shape[0], inputs[j][:, i], color='pink', s=2,
                           label='Branches' if i == 0 and j == 0 else None)

            # Line plot of input trajectory
            ax.plot(time_stamp_inputs, u0_traj[:, i], linestyle='dashed', color='red', label='MPC trajectory')

            ax.set_ylabel(f'Inputs {i + 1}')
            # ax.legend()

        # Set x-axis labels on all plots after looping through inputs and states.
        for ax in axes:
            ax.set_xlabel('Times [s]')
            ax.grid()

        fig.suptitle("CQR State Branch Plots", fontsize=16)

        # Show or return the plot based on `show_plot`
        if show_plot:
            fig.show()
        else:
            plt.close(fig)
            return fig, axes

    def get_robust_input(self, X):

        # init
        labels = X.columns
        input_labels = [col for col in labels if col.endswith('lag_0') and col.startswith('input')]
        state_labels = ([col for col in labels if col.startswith('state')] +
                        [col for col in labels if col.startswith('input') and not col.endswith('lag_0')])

        X_0 = X.iloc[0][state_labels].to_numpy().reshape((-1, 1))
        U_O = X.iloc[0][input_labels].to_numpy().reshape((-1, 1))
        K_lqr = self.get_lqr_gain(X_0, U_O)

        X_0_stacked = np.hstack([X_0] * X.shape[0])
        # K_lqr_stacked = np.vstack([K_lqr] * X.shape[0])

        U_opt_n = (-K_lqr @ (X[state_labels].to_numpy().reshape((-1, self.cqr.n_x)).T - X_0_stacked) +
                   X[input_labels].to_numpy().reshape((-1, self.cqr.n_u)).T)

        X_robust = X.copy()
        # X_robust[input_labels] = U_opt_n
        for i, col in enumerate(input_labels):
            X_robust[col] = U_opt_n[i, :]

        return X_robust

    def get_lqr_gain(self, X_n, U_n):

        # init
        lqr = do_mpc.controller.LQR(model=do_mpc.model.linearize(self.mpc.model, X_n, U_n))

        # setup
        setup_lqr = {'n_horizon': None,
                     't_step': self.mpc.settings.t_step}
        lqr.set_param(**setup_lqr)

        # setting objective
        lqr.set_objective(Q=self.Q, R=self.R)

        # set up lqr
        lqr.setup()

        # end
        return lqr.K

    def make_step(self, x0, enable_plots=False):

        assert x0.shape == (self.cqr.n_x, 1), \
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
        default_ubx = self.bounds_extractor(mpc=self.mpc, bnd_type='upper', var_type='_x')

        default_lbu = self.bounds_extractor(mpc=self.mpc, bnd_type='lower', var_type='_u')
        default_ubu = self.bounds_extractor(mpc=self.mpc, bnd_type='upper', var_type='_u')

        # debug printouts
        if self.verbose:
            print(f"\n\n-------- Iterative Constricting Boundary Nominal Model Predictive Control --------")
            print(f"Time: {self.t0}")

        for i in range(self.max_search):

            # debug printouts
            if self.verbose:
                print(f"\n\n---- Initiating search ----")
                print(f"Time: {self.t0}, Iteration: {i + 1} / {self.max_search}")
                print(f"Frame Number: {self.frame_number}\n\n")

            # re-init to old state
            prev_ic = self._generate_initial_guess(states=prev_state, inputs=prev_input)

            # passing initial cond
            self.mpc.x0 = prev_ic
            self.mpc.set_initial_guess()

            # do make step
            u0 = self.mpc.make_step(x0=x0_current)

            # extracting solver info
            self.store_solver_stats = self.mpc.settings.store_solver_stats
            self.solver_stats = self.mpc.solver_stats

            #if self.store_solver_stats[0] == 'success':
            #    u0 = u0_new
            #else:
            #    break

            lbx = self.bounds_extractor(mpc=self.mpc, bnd_type='lower', var_type='_x')
            ubx = self.bounds_extractor(mpc=self.mpc, bnd_type='upper', var_type='_x')
            lbu = self.bounds_extractor(mpc=self.mpc, bnd_type='lower', var_type='_u')
            ubu = self.bounds_extractor(mpc=self.mpc, bnd_type='upper', var_type='_u')

            boundaries = {}
            boundaries['lbx'] = lbx
            boundaries['ubx'] = ubx
            boundaries['lbu'] = lbu
            boundaries['ubu'] = ubu
            all_boundaries.append(boundaries)

            # extracting optimal trajectories
            u_traj = self.mpc.opt_x_num['_u']
            u_traj_numpy = np.array([entry[0].full().flatten() for entry in u_traj])

            # setting up cqr
            self.cqr.states = current_state
            if self.cqr.order > 1:
                self.cqr.inputs = self.inputs
            self.cqr.set_initial_guess()

            # make branch prediction
            branches = self.make_branch(u0_traj=u_traj_numpy.reshape((-1, self.cqr.n_u)))

            # storage
            if enable_plots:
                all_branches.append(branches)
                plots.append(self.plot_branch_matplotlib(t0=self.t0, show_plot=False))

            # adjust bounds if necessary
            state_adjust_flag = self._adjust_state_bounds(mpc=self.mpc, branches=branches,
                                                          default_lbx=default_lbx, default_ubx=default_ubx)

            input_adjust_flag = self._adjust_input_bounds(mpc=self.mpc, branches=branches,
                                                          default_lbu=default_lbu, default_ubu=default_ubu)

            # increment frame number
            self.frame_number += 1

            if state_adjust_flag or input_adjust_flag:
                adjust_flag = True
            else:
                adjust_flag = False

            if adjust_flag == False:

                # debug printouts
                if self.verbose:
                    print(f"\n\nBoundary adjusted: {adjust_flag}")
                    print(
                        f"modified optimal state upper bound: {self.bounds_extractor(mpc=self.mpc, bnd_type='upper', var_type='_x')}")
                    print(
                        f"modified optimal state lower bound: {self.bounds_extractor(mpc=self.mpc, bnd_type='lower', var_type='_x')}")
                    print(
                        f"modified optimal input upper bound: {self.bounds_extractor(mpc=self.mpc, bnd_type='upper', var_type='_u')}")
                    print(
                        f"modified optimal input lower bound: {self.bounds_extractor(mpc=self.mpc, bnd_type='lower', var_type='_u')}")
                    print(f"Time: {self.t0}, Iteration: {i + 1} / {self.max_search}")
                    print(f"surrogate predicted optimal input: {u0}")
                    print(f"surrogate predicted optimal next state: {branches['states'][1][0, :]}")
                    print(f"initial state: {branches['states'][0]}")
                    print()
                    print('Printing branches form cqr')
                    print(self.cqr.branches['u0_traj'])
                    print(f"-------- Success! Feasible input found. --------\n\n")

                break

            elif i == self.max_search - 1:

                # debug printouts
                if self.verbose:
                    print(f"\n\nBoundary adjusted: {adjust_flag}")
                    print(
                        f"last modified state upper bound: {self.bounds_extractor(mpc=self.mpc, bnd_type='upper', var_type='_x')}")
                    print(
                        f"last modified state lower bound: {self.bounds_extractor(mpc=self.mpc, bnd_type='lower', var_type='_x')}")
                    print(
                        f"last modified input upper bound: {self.bounds_extractor(mpc=self.mpc, bnd_type='upper', var_type='_u')}")
                    print(
                        f"last modified input lower bound: {self.bounds_extractor(mpc=self.mpc, bnd_type='lower', var_type='_u')}")
                    print(f"Time: {self.t0}, Iteration: {i + 1} / {self.max_search}")
                    print(f"surrogate predicted input: {u0}")
                    print(f"surrogate predicted next state: {branches['states'][1][0, :]}")
                    print(f"initial state: {branches['states'][0]}")
                    print()
                    print('Printing branches form cqr')
                    print(self.cqr.branches['u0_traj'])
                    print(
                        f"-------- Max interation reached and feasible input not found! Last calculated input returned. --------\n\n")

            else:

                # debug printouts
                if self.verbose:
                    print(f"\n\nBoundary adjusted: {adjust_flag}")
                    print(
                        f"re-modified state upper bound: {self.bounds_extractor(mpc=self.mpc, bnd_type='upper', var_type='_x')}")
                    print(
                        f"re-modified state lower bound: {self.bounds_extractor(mpc=self.mpc, bnd_type='lower', var_type='_x')}")
                    print(
                        f"re-modified input upper bound: {self.bounds_extractor(mpc=self.mpc, bnd_type='upper', var_type='_u')}")
                    print(
                        f"re-modified input lower bound: {self.bounds_extractor(mpc=self.mpc, bnd_type='lower', var_type='_u')}")
                    print(f"Time: {self.t0}, Iteration: {i + 1} / {self.max_search}")
                    print(f"surrogate predicted input: {u0}")
                    print(f"surrogate predicted next state: {branches['states'][1][0, :]}")
                    print(f"initial state: {branches['states'][0]}")
                    print()
                    print('Printing branches form cqr')
                    print(self.cqr.branches['u0_traj'])
                    print(
                        f"---- Feasible input not found! Recalculating again. ---->>>>\n\n")

        # revert back to the system boundaries for the state
        self.bounds_setter(mpc=self.mpc, bnd_type='upper', var_type='_x', bnd_val=default_ubx)
        self.bounds_setter(mpc=self.mpc, bnd_type='lower', var_type='_x', bnd_val=default_lbx)

        self.bounds_setter(mpc=self.mpc, bnd_type='upper', var_type='_u', bnd_val=default_ubu)
        self.bounds_setter(mpc=self.mpc, bnd_type='lower', var_type='_u', bnd_val=default_lbu)

        # push the new u0 into the new initial condition
        if self.cqr.order > 1:
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

    def _adjust_state_bounds(self, mpc, branches, default_lbx, default_ubx):
        # checking if all the predicted states saty inside the boundary
        # for i in range(all_states.shape[0]):
        #    current_state = all_states[i, :].reshape((1, -1))
        #    if (np.any(current_state < lbx.reshape((-1,))[0: self.cqr.n_x]) or
        #            np.any(current_state > ubx.reshape((-1,))[0: self.cqr.n_x])):
        #        adjust_flag = True
        #        break

        if len(branches['states']) > 1:

            all_states = np.vstack(branches['states'][1:])

            default_lbx_matrix = np.vstack([default_lbx.reshape((-1,))[0: self.cqr.n_x]] * all_states.shape[0])
            default_ubx_matrix = np.vstack([default_ubx.reshape((-1,))[0: self.cqr.n_x]] * all_states.shape[0])

            if (np.any(all_states < default_lbx_matrix) or np.any(all_states > default_ubx_matrix)):
                adjust_flag = True

                # calculation per alpha value
                for i, states_n in enumerate(branches['states']):

                    if i == 0:
                        continue

                    # init
                    default_lbx_matrix_n = np.vstack([default_lbx.reshape((-1,))[0: self.cqr.n_x]] * states_n.shape[0])
                    default_ubx_matrix_n = np.vstack([default_ubx.reshape((-1,))[0: self.cqr.n_x]] * states_n.shape[0])
                    default_range_x_n = default_ubx_matrix_n - default_lbx_matrix_n
                    # alpha_n = branches['alphas'][i]

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
                    # upper_prob = alpha_n * residue_prob_u
                    # lower_prob = alpha_n * residue_prob_l
                    upper_prob_stacked = np.vstack([residue_prob_u] * states_n.shape[0])
                    lower_prob_stacked = np.vstack([residue_prob_l] * states_n.shape[0])

                    if i == 1:
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
                assert np.all(new_lbx < new_ubx), ("Lower bound has to be is greater than upper bound! "
                                                   "Reduce 'tightner' to ensure this does not occur.")

                # readjusting boundaries
                self.bounds_setter(mpc=mpc, bnd_type='upper', var_type='_x', bnd_val=new_ubx)
                self.bounds_setter(mpc=mpc, bnd_type='lower', var_type='_x', bnd_val=new_lbx)

                # resetting history
                self.mpc.reset_history()

            else:
                adjust_flag = False

        else:
            adjust_flag = False

        return adjust_flag

    def _adjust_input_bounds(self, mpc, branches, default_lbu, default_ubu):
        # checking if all the predicted states saty inside the boundary
        # for i in range(all_states.shape[0]):
        #    current_state = all_states[i, :].reshape((1, -1))
        #    if (np.any(current_state < lbx.reshape((-1,))[0: self.cqr.n_x]) or
        #            np.any(current_state > ubx.reshape((-1,))[0: self.cqr.n_x])):
        #        adjust_flag = True
        #        break

        if len(branches['inputs']) > 1:

            all_states = np.vstack(branches['inputs'][1:])

            default_lbu_matrix = np.vstack([default_lbu.reshape((-1,))[0: self.cqr.n_x]] * all_states.shape[0])
            default_ubu_matrix = np.vstack([default_ubu.reshape((-1,))[0: self.cqr.n_x]] * all_states.shape[0])

            if (np.any(all_states < default_lbu_matrix) or np.any(all_states > default_ubu_matrix)):
                adjust_flag = True

                # calculation per alpha value
                for i, states_n in enumerate(branches['inputs']):

                    if i == 0:
                        continue

                    # init
                    default_lbu_matrix_n = np.vstack([default_lbu.reshape((-1,))[0: self.cqr.n_x]] * states_n.shape[0])
                    default_ubu_matrix_n = np.vstack([default_ubu.reshape((-1,))[0: self.cqr.n_x]] * states_n.shape[0])
                    default_range_u_n = default_ubu_matrix_n - default_lbu_matrix_n
                    # alpha_n = branches['alphas'][i]

                    # checking the protrusions above the upper boundary, if positive: boundary is crossed
                    residue_upper = (states_n - default_ubu_matrix_n) / default_range_u_n

                    # checking the protrusions below the lower boundary, if positive: boundary is crossed
                    residue_lower = (- states_n + default_lbu_matrix_n) / default_range_u_n

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
                    # upper_prob = alpha_n * residue_prob_u
                    # lower_prob = alpha_n * residue_prob_l
                    upper_prob_stacked = np.vstack([residue_prob_u] * states_n.shape[0])
                    lower_prob_stacked = np.vstack([residue_prob_l] * states_n.shape[0])

                    if i == 1:
                        prob_scaled_states_u = upper_prob_stacked * residue_upper
                        prob_scaled_states_l = lower_prob_stacked * residue_lower

                    else:
                        prob_scaled_states_u = np.vstack([prob_scaled_states_u, upper_prob_stacked * residue_upper])
                        prob_scaled_states_l = np.vstack([prob_scaled_states_l, lower_prob_stacked * residue_lower])

                # determining the mean of the deviations
                prob_upper = np.mean(prob_scaled_states_u, axis=0)
                prob_lower = np.mean(prob_scaled_states_l, axis=0)

                # extracting current mpc bounds
                lbu = self.bounds_extractor(mpc=mpc, bnd_type='lower', var_type='_u')
                ubu = self.bounds_extractor(mpc=mpc, bnd_type='upper', var_type='_u')

                range_u = ubu - lbu
                range_u = range_u[0:self.cqr.n_x, :]

                # adjusting the bounds
                adj_lower = self.tightner * prob_lower * range_u.reshape((-1,))
                adj_upper = self.tightner * prob_upper * range_u.reshape((-1,))

                new_lbu = lbu[0:self.cqr.n_x, :] + adj_lower.reshape((-1, 1))
                new_ubu = ubu[0:self.cqr.n_x, :] - adj_upper.reshape((-1, 1))

                # sanity check
                assert np.all(new_lbu < new_ubu), ("Lower bound has to be is greater than upper bound! "
                                                   "Reduce 'tightner' to ensure this does not occur.")

                # readjusting boundaries
                self.bounds_setter(mpc=mpc, bnd_type='upper', var_type='_u', bnd_val=new_ubu)
                self.bounds_setter(mpc=mpc, bnd_type='lower', var_type='_u', bnd_val=new_lbu)

                # resetting history
                self.mpc.reset_history()

            else:
                adjust_flag = False

        else:
            adjust_flag = False

        return adjust_flag

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
            lbu = boundaries['lbu'].reshape(-1, )
            ubu = boundaries['ubu'].reshape(-1, )

            branch_times = history['time'] + [num + history['time'][-1] for num in branches['time_stamps']]

            fig, axes = plots[k]

            # Loop through each state variable (n_x)
            for i in range(n_x):
                ax = axes[i]

                # Simulation line (history of states)
                ax.plot(history['time'], history['x0'][i, :], color='#1f77b4', linestyle='solid',
                        label='Simulation' if i == 0 else None)

                # System bounds (upper and lower)
                #ax.plot(branch_times, [self.cqr.ubx[i]] * len(branch_times), color='black', linestyle='solid',
                #        label='System Bounds' if i == 0 else None)
                #ax.plot(branch_times, [self.cqr.lbx[i]] * len(branch_times), color='black', linestyle='solid')

                # System bounds (upper and lower)
                upper_limit = np.full((len(branch_times),), self.cqr.ubx[i])
                lower_limit = np.full((len(branch_times),), self.cqr.lbx[i])

                # gray infill
                ax.fill_between(branch_times, lower_limit, upper_limit, color='gray',
                                  alpha=0.5, label='System Bounds' if i == 0 else None)

                # Optimized MPC bounds (upper and lower)
                ax.plot(branch_times, [ubx[i]] * len(branch_times), color='purple', linestyle='dashed',
                        label='MPC Bound' if i == 0 else None)
                ax.plot(branch_times, [lbx[i]] * len(branch_times), color='purple', linestyle='dashed')

                # ax.set_ylabel(f'State {i + 1}')

            # Loop through each control variable (n_u)
            for i in range(n_u):
                ax = axes[n_x + i]

                # Combine historical control inputs with the first value from branches
                u_combined = np.hstack([history['u0'][i, :-1], branches['u0_traj'][0, i]])

                ax.plot(history['time'], u_combined, color='#1f77b4', linestyle='solid')

                # System bounds (upper and lower)
                #ax.plot(branch_times, [self.cqr.ubu[i]] * len(branch_times), color='black', linestyle='solid')
                #ax.plot(branch_times, [self.cqr.lbu[i]] * len(branch_times), color='black', linestyle='solid')

                # System bounds (upper and lower)
                upper_limit = np.full((len(branch_times),), self.cqr.ubu[i])
                lower_limit = np.full((len(branch_times),), self.cqr.lbu[i])

                # gray infill
                ax.fill_between(branch_times, lower_limit, upper_limit, color='gray',
                                alpha=0.5)

                # Optimized MPC bounds (upper and lower)
                ax.plot(branch_times, [ubu[i]] * len(branch_times), color='purple', linestyle='dashed')
                ax.plot(branch_times, [lbu[i]] * len(branch_times), color='purple', linestyle='dashed')

                # ax.set_ylabel(f'Input {j + 1}')

            # Set x-axis labels on all plots after looping through inputs and states.
            # for ax in axes:
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