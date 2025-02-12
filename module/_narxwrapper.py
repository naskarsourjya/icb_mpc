import numpy as np
import do_mpc

class mpc_narx(do_mpc.controller.MPC):
    def __init__(self, model, order, n_x, n_u):


        assert order >= 1, "Order has to be a finite positive integer more than or equal to 1."

        # storage
        super(mpc_narx, self).__init__(model=model)     # `self` is the do_mpc MPC Controller
        self.narx_order = order
        self.narx_n_x = n_x
        self.narx_n_u = n_u
        self._narxstates = None
        self._narxinputs = None

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

        assert val.shape[1] == self.narx_order, \
            'Number of samples must be equal to the order of the NARX model!'

        assert val.shape[0] == self.narx_n_x, (
            'Expected number of states is: {}, but found {}'.format(self.narx_n_x, val.shape[0]))

        # storage
        self._narxstates = val


    @property
    def inputs(self):
        return self._narxinputs


    @inputs.setter
    def inputs(self, val):
        if self.narx_order>1:
            assert isinstance(val, np.ndarray), "inputs must be a numpy.array."

            assert self.narx_order - 1 == val.shape[1], \
                'Number of samples for inputs should be (order-1) !'

            assert val.shape[0] == self.narx_n_u, (
                'Expected number of inputs is: {}, but found {}'.format(self.narx_n_u, val.shape[0]))

            # storage
            self._narxinputs = val

        # error
        else:
            raise ValueError("Inputs cannot be set for system with order <= 1.")

    def _generate_initial_guess(self):

        # init
        states = self.states
        inputs = self.inputs

        order = self.narx_order
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





    def narx_set_initial_guess(self):

        # calculate initial guess
        initial_cond = self._generate_initial_guess()

        # passing initial cond
        self.x0 = initial_cond
        self.set_initial_guess()


        #self.mpc['history'] = None

        # end
        return None

    def narx_make_step(self, x0):

        assert x0.shape==(self.narx_n_x, 1), \
            f"x0 should have shape ({self.narx_n_x}, 1). Shape found instead is: {x0.shape}"

        # init
        initial_cond = self.initial_cond
        n_x = self.narx_n_x
        n_u = self.narx_n_u
        order = self.narx_order

        # segregating states and inputs
        states = initial_cond[0:n_x * order, :]
        inputs = initial_cond[n_x * order:, :]

        # stacking current state
        x_next = np.vstack([x0, states[0:n_x * (order - 1)], inputs])

        # determining optimal input
        u0 = self.make_step(x0=x_next)

        # shifitng initial condition
        next_state_history = np.vstack([x0, states[0:n_x * (order - 1)]])
        next_input_history = np.vstack([u0, inputs[n_u:]])

        # pushing it to class
        self.states = self.reshape(next_state_history, shape=(n_x, -1))
        self.inputs = self.reshape(next_input_history, shape=(n_u, -1))
        self._generate_initial_guess()

        # end
        return u0

