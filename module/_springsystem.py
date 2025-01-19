import do_mpc
import numpy as np
import casadi as ca

class SpringSystem:
    def __init__(self, set_seed = True, suppress_ipopt = True):
        # for repetable results
        if set_seed:
            np.random.seed(0)
        self.suppress_ipopt = suppress_ipopt
        
        # hyperparameters
        self._set_hyperparameters()

        # setting up sysetm
        #self.model= self._get_spring_model()
        #self.simulator = self._get_spring_simulator(model=  self.model)
        #self.random_traj_mpc = self._get_spring_random_traj_mpc(model= self.model)
        #self.estimator = do_mpc.estimator.StateFeedback(model= self.model)

        return None

    def _set_hyperparameters(self):
        self.t_step = 0.1
        self.n_horizon = 10
        self.r = 0.25   # penalty for input
                        # range betn [0 -1]
                        # increasing this reduces oscillatroy behaviou of input
        # box constraints
        self.lbx = np.array([-1, -1])   # [lower_bound_position, lower_bound_velocity]
        self.ubx = np.array([1, 1])     # [upper_bound_position, upper_bound_velocity]
        self.lbu = np.array([-1])       # [lower_bound_f_ext]
        self.ubu = np.array([1])        # [upper_bound_f_ext]

        return None

    def _get_model(self):
        # init
        model = do_mpc.model.Model('continuous', symvar_type='SX')

        # Define the states
        position = model.set_variable(var_type='_x', var_name='position', shape=(1,1))
        velocity = model.set_variable(var_type='_x', var_name='velocity', shape=(1,1))

        position_ref = model.set_variable(var_type='_tvp', var_name='position_ref', shape=(1,1))
        velocity_ref = model.set_variable(var_type='_tvp', var_name='velocity_ref', shape=(1,1))

        # Define the control inputs
        f_external = model.set_variable(var_type='_u', var_name='f_external', shape=(1,1))

        # constants
        k = 1.0     # spring constant
        c = 0.1     # damping constant
        mass = 1.0  # mass of the object

        # Define the model equations
        model.set_rhs('position', velocity)
        model.set_rhs('velocity', (-k*position - c*velocity + f_external)/mass)

        # to generate random setpoints
        model.set_expression('position_ref', position_ref)
        model.set_expression('velocity_ref', velocity_ref)

        model.setup()

        # end
        return model

    def _get_simulator(self, model):
        # init
        simulator = do_mpc.simulator.Simulator(model)
        
        # set t_step
        simulator.set_param(t_step = self.t_step)

        # tvp
        tvp_template = simulator.get_tvp_template()

        def tvp_fun(t_ind):
            return tvp_template

        simulator.set_tvp_fun(tvp_fun)
        
        # simulator setup
        simulator.setup()

        # end
        return simulator

    def _get_random_traj_mpc(self, model):
        # init
        mpc = do_mpc.controller.MPC(model)

        # supperess ipopt output
        if self.suppress_ipopt:
            mpc.settings.supress_ipopt_output()
        
        # set t_step
        mpc.set_param(t_step = self.t_step)
        
        # set horizon
        mpc.set_param(n_horizon = self.n_horizon)

        # setting up cost function
        mterm = (1-self.r)*((model.x['position'] - model.tvp['position_ref'])**2
                 +(model.x['velocity'] - model.tvp['velocity_ref'])**2)

        mpc.set_objective(mterm=mterm, lterm=0*mterm)

        mpc.set_rterm(f_external=self.r)
        
        # set control bounds
        mpc.bounds['lower','_u','f_external'] = self.lbu
        mpc.bounds['upper','_u','f_external'] = self.ubu
        
        # set state bounds
        mpc.bounds['lower','_x','position'] = self.lbx[0]
        mpc.bounds['upper','_x','position'] = self.ubx[0]
        mpc.bounds['lower','_x','velocity'] = self.lbx[1]
        mpc.bounds['upper','_x','velocity'] = self.ubx[1]

        # enter random setpoints inside the box constraints
        tvp_template = mpc.get_tvp_template()

        # sending random setpoints inside box constraints
        def tvp_fun(t_ind):
            x_ref = np.random.uniform(self.lbx, self.ubx)
            #print(x_ref)
            tvp_template['_tvp', :, 'position_ref'] = x_ref[0]
            tvp_template['_tvp', :, 'velocity_ref'] = x_ref[1]
            return tvp_template

        mpc.set_tvp_fun(tvp_fun)
        
        # setup
        mpc.setup()
        
        # end
        return mpc
