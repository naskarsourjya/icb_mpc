import numpy as np
from tqdm import tqdm


class MPC_Brancher():
    def __init__(self, mpc, cqr, tightner = 0.1):
        # storage
        #super(mpc_narx, self).__init__(model=model)     # `self` is the do_mpc MPC Controller
        self.mpc = mpc
        self.cqr = cqr
        self._narxstates = None
        self._narxinputs = None
        self.tightner = tightner

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

    def make_step(self, x0, max_iter=10):

        assert x0.shape==(self.cqr.n_x, 1), \
            f"x0 should have shape ({self.cqr.n_x}, 1). Shape found instead is: {x0.shape}"

        # init
        initial_cond = self.initial_cond
        n_x = self.cqr.n_x
        n_u = self.cqr.n_u
        order = self.cqr.order
        narx_state_length = order * n_x + (order - 1) * n_u
        prev_ubx = self.mpc.bounds['upper', '_x', 'system_state']
        prev_lbx = self.mpc.bounds['lower', '_x', 'system_state']
        prev_input = self.inputs
        prev_state = self.states

        # segregating states and inputs
        states = initial_cond[0:n_x * order, :]
        inputs = initial_cond[n_x * order:, :]

        # stacking current state
        x_next = np.vstack([x0, states[0:n_x * (order - 1)], inputs])

        # determining optimal input
        for i in range(max_iter):
            # init
            adjust_falg = False
            self.inputs = prev_input
            self.states = prev_state
            self.set_initial_guess()

            # making mpc prediction
            u0 = self.mpc.make_step(x0=x_next)

            # extracting optimal trajectories
            u_traj = self.mpc.opt_x_num['_u']
            x_traj = self.mpc.opt_x_num['_x']

            # simulating cqr with this optimal trajectory
            x_traj_numpy = np.array([entry[0][0].full().flatten() for entry in x_traj]).T
            u_traj_numpy = np.array([entry[0][0].full().flatten() for entry in u_traj]).T

            # setting up cqr
            self.cqr.states = self.states
            self.cqr.inputs = self.inputs
            self.cqr.set_initial_guess()

            # extraction of mpc boundaries
            ubx = self.mpc.bounds['upper', '_x', 'system_state']
            lbx = self.mpc.bounds['lower', '_x', 'system_state']

            # make branch prediction
            results = self.cqr.make_branch(u0_traj = u_traj_numpy,
                                           lbx=lbx.full()[0: self.cqr.n_x, :].reshape((-1,)),
                                           ubx=ubx.full()[0: self.cqr.n_x, :].reshape((-1,)))

            # reading the results
            all_states = np.hstack(results['states'])

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
        next_state_history = np.vstack([x0, states[0:n_x * (order - 1)]])
        next_input_history = np.vstack([u0, inputs[n_u:]])

        # pushing it to class
        self.states = self.reshape(next_state_history, shape=(n_x, -1))
        self.inputs = self.reshape(next_input_history, shape=(n_u, -1))
        self._generate_initial_guess()

        # end
        return u0

