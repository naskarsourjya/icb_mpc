import torch
import do_mpc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt


class MPC_NARX():
    def __init__(self, mpc, n_x, n_u, order, verbose=True):
        
        # storage
        #super(mpc_narx, self).__init__(model=model)     # `self` is the do_mpc MPC Controller
        self.mpc = mpc
        self.n_x = n_x
        self.n_u = n_u
        self.order = order
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

        assert val.shape[0] == self.order, \
            'Number of samples must be equal to the order of the NARX model!'

        assert val.shape[1] == self.n_x, (
            'Expected number of states is: {}, but found {}'.format(self.n_x, val.shape[0]))

        # storage
        self._narxstates = val


    @property
    def inputs(self):
        return self._narxinputs


    @inputs.setter
    def inputs(self, val):
        if self.order>1:
            assert isinstance(val, np.ndarray), "inputs must be a numpy.array."

            assert self.order - 1 == val.shape[0], \
                'Number of samples for inputs should be (order-1) !'

            assert val.shape[1] == self.n_u, (
                'Expected number of inputs is: {}, but found {}'.format(self.n_u, val.shape[0]))

            # storage
            self._narxinputs = val

        # error
        else:
            raise ValueError("Inputs cannot be set for system with order <= 1.")


    def _generate_initial_guess(self, states, inputs=None):

        assert isinstance(states, np.ndarray), "states must be a numpy.array."

        assert states.shape[0] == self.order, \
            'Number of samples must be equal to the order of the NARX model!'

        assert states.shape[1] == self.n_x, (
            'Expected number of states is: {}, but found {}'.format(self.n_x, states.shape[0]))

        if self.order>1:

            assert isinstance(inputs, np.ndarray), "inputs must be a numpy.array."

            assert self.order - 1 == inputs.shape[0], \
                'Number of samples for inputs should be (order-1) !'

            assert inputs.shape[1] == self.n_u, (
                'Expected number of inputs is: {}, but found {}'.format(self.n_u, inputs.shape[0]))

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


    def make_step(self, x0):

        assert x0.shape == (self.n_x, 1), \
            f"x0 should have shape ({self.n_x}, 1). Shape found instead is: {x0.shape}"

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
        if self.order > 1:
            pseudo_input_history = np.vstack([u0.reshape((1, -1)),
                                              self.inputs[:-1, :]])
            self.inputs = pseudo_input_history
        self.states = current_state

        # end
        return u0