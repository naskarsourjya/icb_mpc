import do_mpc
import numpy as np
import casadi as ca


class CSTR_dompc:
    def __init__(self, set_seed=None, suppress_ipopt=True):
        # for repetable results
        if set_seed is not None:
            np.random.seed(set_seed)

        self.set_seed = set_seed
        self.suppress_ipopt = suppress_ipopt

        # hyperparameters
        self._set_hyperparameters()

        # setting up sysetm
        # self.model= self._get_spring_model()
        # self.simulator = self._get_spring_simulator(model=  self.model)
        # self.random_traj_mpc = self._get_spring_random_traj_mpc(model= self.model)
        # self.estimator = do_mpc.estimator.StateFeedback(model= self.model)

        # end of init

    def _set_hyperparameters(self):

        self.t_step = 0.005
        # range betn [0 -1]
        # increasing this reduces oscillatroy behaviou of input
        # box constraints
        self.lbx = np.array([0.1, 0.1, 50, 50])  # [lower_bound_position, lower_bound_velocity]
        self.ubx = np.array([2, 2, 140, 140])  # [upper_bound_position, upper_bound_velocity]
        self.lbu = np.array([5, -8500])  # [lower_bound_f_ext]
        self.ubu = np.array([100, 0])  # [upper_bound_f_ext]

        return None

    def _get_model(self):

        # init
        model_type = 'continuous'  # either 'discrete' or 'continuous'
        model = do_mpc.model.Model(model_type, 'SX')

        # Certain parameters
        K0_ab = 1.287e12  # K0 [h^-1]
        K0_bc = 1.287e12  # K0 [h^-1]
        K0_ad = 9.043e9  # K0 [l/mol.h]
        R_gas = 8.3144621e-3  # Universal gas constant
        E_A_ab = 9758.3 * 1.00  # * R_gas# [kj/mol]
        E_A_bc = 9758.3 * 1.00  # * R_gas# [kj/mol]
        E_A_ad = 8560.0 * 1.0  # * R_gas# [kj/mol]
        H_R_ab = 4.2  # [kj/mol A]
        H_R_bc = -11.0  # [kj/mol B] Exothermic
        H_R_ad = -41.85  # [kj/mol A] Exothermic
        Rou = 0.9342  # Density [kg/l]
        Cp = 3.01  # Specific Heat capacity [kj/Kg.K]
        Cp_k = 2.0  # Coolant heat capacity [kj/kg.k]
        A_R = 0.215  # Area of reactor wall [m^2]
        V_R = 10.01  # 0.01 # Volume of reactor [l]
        m_k = 5.0  # Coolant mass[kg]
        T_in = 130.0  # Temp of inflow [Celsius]
        K_w = 4032.0  # [kj/h.m^2.K]
        C_A0 = (5.7 + 4.5) / 2.0 * 1.0  # Concentration of A in input Upper bound 5.7 lower bound 4.5 [mol/l]

        # States struct (optimization variables):
        C_a = model.set_variable(var_type='_x', var_name='C_a', shape=(1, 1))
        C_b = model.set_variable(var_type='_x', var_name='C_b', shape=(1, 1))
        T_R = model.set_variable(var_type='_x', var_name='T_R', shape=(1, 1))
        T_K = model.set_variable(var_type='_x', var_name='T_K', shape=(1, 1))

        # Input struct (optimization variables):
        F = model.set_variable(var_type='_u', var_name='F')
        Q_dot = model.set_variable(var_type='_u', var_name='Q_dot')

        # Fixed parameters:
        alpha = model.set_variable(var_type='_p', var_name='alpha')
        beta = model.set_variable(var_type='_p', var_name='beta')

        # Set expression. These can be used in the cost function, as non-linear constraints
        # or just to monitor another output.
        T_dif = model.set_expression(expr_name='T_dif', expr=T_R - T_K)

        # Expressions can also be formed without being explicitly added to the model.
        # The main difference is that they will not be monitored and can only be used within the current file.
        K_1 = beta * K0_ab * ca.exp((-E_A_ab) / ((T_R + 273.15)))
        K_2 = K0_bc * ca.exp((-E_A_bc) / ((T_R + 273.15)))
        K_3 = K0_ad * ca.exp((-alpha * E_A_ad) / ((T_R + 273.15)))

        # Differential equations
        model.set_rhs('C_a', F * (C_A0 - C_a) - K_1 * C_a - K_3 * (C_a ** 2))
        model.set_rhs('C_b', -F * C_b + K_1 * C_a - K_2 * C_b)
        model.set_rhs('T_R',
                      ((K_1 * C_a * H_R_ab + K_2 * C_b * H_R_bc + K_3 * (C_a ** 2) * H_R_ad) / (-Rou * Cp)) + F * (
                                  T_in - T_R) + (((K_w * A_R) * (-T_dif)) / (Rou * Cp * V_R)))
        model.set_rhs('T_K', (Q_dot + K_w * A_R * (T_dif)) / (m_k * Cp_k))

        # Build the model
        model.setup()

        # end
        return model

    def _get_simulator(self, model):

        # init
        simulator = do_mpc.simulator.Simulator(model)

        # setting up parameters for the simulator
        params_simulator = {
            'integration_tool': 'cvodes',
            'abstol': 1e-10,
            'reltol': 1e-10,
            't_step': self.t_step
        }
        simulator.set_param(**params_simulator)

        # setting up time varying parameters (tvp)
        tvp_num = simulator.get_tvp_template()

        def tvp_fun(t_now):
            return tvp_num

        simulator.set_tvp_fun(tvp_fun)

        # setting up parameters for the simulator
        p_num = simulator.get_p_template()
        p_num['alpha'] = 1
        p_num['beta'] = 1

        def p_fun(t_now):
            return p_num

        simulator.set_p_fun(p_fun)

        # completing the simulator setup
        simulator.setup()

        # end
        return simulator

    def _get_mpc(self, model, n_horizon, setpoint, r=1):
        # init
        mpc = do_mpc.controller.MPC(model)

        # Set settings of MPC:
        mpc.settings.n_horizon = n_horizon
        mpc.settings.n_robust = 1
        mpc.settings.open_loop = 0
        mpc.settings.t_step = self.t_step
        mpc.settings.state_discretization = 'collocation'
        mpc.settings.collocation_type = 'radau'
        mpc.settings.collocation_deg = 2
        mpc.settings.collocation_ni = 1
        mpc.settings.store_full_solution = True

        # suppress solver output
        if self.suppress_ipopt:
            mpc.settings.supress_ipopt_output()

        # setting up the scaling of the variables
        mpc.scaling['_x', 'T_R'] = 100
        mpc.scaling['_x', 'T_K'] = 100
        mpc.scaling['_u', 'Q_dot'] = 2000
        mpc.scaling['_u', 'F'] = 100

        # setting up the cost function
        mterm = (model.x['C_b'] - setpoint) ** 2
        #lterm = (model.x['C_b'] - 0.6) ** 2
        mpc.set_objective(mterm=mterm, lterm=mterm)

        # setting up the factors for input penalisation
        mpc.set_rterm(F=r*0.1, Q_dot=r*1e-3)

        # setting up lower boundaries for the states
        mpc.bounds['lower', '_x', 'C_a'] = 0.1
        mpc.bounds['lower', '_x', 'C_b'] = 0.1
        mpc.bounds['lower', '_x', 'T_R'] = 50
        mpc.bounds['lower', '_x', 'T_K'] = 50

        # setting up upper boundaries for the states
        mpc.bounds['upper', '_x', 'C_a'] = 2
        mpc.bounds['upper', '_x', 'C_b'] = 2
        mpc.bounds['upper', '_x', 'T_K'] = 140

        # setting up lower boundaries for the inputs
        mpc.bounds['lower', '_u', 'F'] = 5
        mpc.bounds['lower', '_u', 'Q_dot'] = -8500

        # setting up upper boundaries for the inputs
        mpc.bounds['upper', '_u', 'F'] = 100
        mpc.bounds['upper', '_u', 'Q_dot'] = 0.0

        # Instead of having a regular bound on T_R:
        # mpc.bounds['upper', '_x', 'T_R'] = 140
        # We can also have soft constraints as part of the set_nl_cons method:
        mpc.set_nl_cons('T_R', model.x['T_R'], ub=140, soft_constraint=True, penalty_term_cons=1e2)

        # setting up parameter uncertainty
        alpha_var = np.array([1., 1.05, 0.95])
        beta_var = np.array([1., 1.1, 0.9])
        mpc.set_uncertainty_values(alpha=alpha_var, beta=beta_var)

        # completing the setup of the mpc
        mpc.setup()

        # end
        return mpc
