import do_mpc
import numpy as np
import casadi as ca


class CSTR:
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
        self.t_step = 0.1
        # range betn [0 -1]
        # increasing this reduces oscillatroy behaviou of input
        # box constraints
        self.lbx = np.array([0, 0, 145, 120])  # [lower_bound_position, lower_bound_velocity]
        self.ubx = np.array([5, 5, 150, 170])  # [upper_bound_position, upper_bound_velocity]
        self.lbu = np.array([5])  # [lower_bound_f_ext]
        self.ubu = np.array([35])  # [upper_bound_f_ext]

        return None

    def _get_model(self):
        # init
        model = do_mpc.model.Model('continuous', symvar_type='SX')

        # Certain parameters
        k0_1 = 1.287e12  # K0 [h^-1]
        k0_2 = 1.287e12  # K0 [h^-1]
        k0_3 = 9.043e9  # K0 [l/mol.h]
        R_gas = 8.3144621e-3  # Universal gas constant
        E_A_1 = 9758.3 * R_gas# [kj/mol]
        E_A_2 = 9758.3 * R_gas# [kj/mol]
        E_A_3 = 7704.0 * R_gas# [kj/mol]
        H_R_ab = 4.2  # [kj/mol A]
        H_R_bc = -11.0  # [kj/mol B] Exothermic
        H_R_ad = -41.85  # [kj/mol A] Exothermic
        rho = 0.9342  # Density [kg/l]
        Cp = 3.01  # Specific Heat capacity [kj/Kg.K]
        Cp_k = 2.0  # Coolant heat capacity [kj/kg.k]
        A = 0.215  # Area of reactor wall [m^2]
        V_R = 10.01  # 0.01 # Volume of reactor [l]
        m_k = 5.0  # Coolant mass[kg]
        T_in = 130.0  # Temp of inflow [Celsius]
        k_w = 4032.0  # [kj/h.m^2.K]
        Q_dot_K = -4500 # [kJ h^-1]
        c_A0 = 0.8 # [mol l^-1]

        # states
        c_A = model.set_variable(var_type='_x', var_name='c_A', shape=(1, 1))
        c_B = model.set_variable(var_type='_x', var_name='c_B', shape=(1, 1))
        T_R = model.set_variable(var_type='_x', var_name='T_R', shape=(1, 1))
        T_j = model.set_variable(var_type='_x', var_name='T_j', shape=(1, 1))

        # input
        F = model.set_variable(var_type='_u', var_name='F', shape=(1, 1))

        # extra equation
        k_1 = k0_1 * ca.exp(-E_A_1/(R_gas*(T_R+273.15)))
        k_2 = k0_2 * ca.exp(-E_A_2/(R_gas*(T_R+273.15)))
        k_3 = k0_3 * ca.exp(-E_A_3 / (R_gas * (T_R + 273.15)))

        # setting rhs
        model.set_rhs('c_A', F*(c_A0 - c_A) - k_1*c_A - k_3*(c_A ** 2))
        model.set_rhs('c_B', -F*c_B + k_1*c_A - k_2*c_B)
        model.set_rhs('T_R', -F*(T_in - T_R) + k_w*A*(T_j - T_R)/(rho*Cp*V_R)
                      - (k_1*c_A*H_R_ab + k_2*c_B*H_R_bc + k_3*c_A*c_A*H_R_ad)/(rho*Cp))
        model.set_rhs('T_j', (Q_dot_K + k_w*A*(T_R-T_j))/(m_k*Cp_k))

        # setup
        model.setup()

        # end
        return model

    def _get_simulator(self, model):
        # init
        simulator = do_mpc.simulator.Simulator(model)

        # set t_step
        simulator.set_param(t_step=self.t_step)

        # tvp
        tvp_template = simulator.get_tvp_template()

        def tvp_fun(t_ind):
            return tvp_template

        simulator.set_tvp_fun(tvp_fun)

        # simulator setup
        simulator.setup()

        # end
        return simulator

    def _get_mpc(self, model, n_horizon, setpoint, r):
        # init
        mpc = do_mpc.controller.MPC(model)

        # supperess ipopt output
        if self.suppress_ipopt:
            mpc.settings.supress_ipopt_output()

        # set t_step
        mpc.set_param(t_step=self.t_step)

        # set horizon
        mpc.set_param(n_horizon=n_horizon)

        # setting up cost function
        x = model.x.master
        mterm = (setpoint - x[0]) ** 2

        # passing objective function
        mpc.set_objective(mterm=mterm, lterm=mterm)

        # input penalisation
        mpc.set_rterm(F=r)

        # set control bounds
        mpc.bounds['lower', '_u', 'F'] = self.lbu
        mpc.bounds['upper', '_u', 'F'] = self.ubu

        # set state bounds
        mpc.bounds['lower', '_x', 'c_A'] = self.lbx[0]
        mpc.bounds['upper', '_x', 'c_A'] = self.ubx[0]
        mpc.bounds['lower', '_x', 'c_B'] = self.lbx[1]
        mpc.bounds['upper', '_x', 'c_B'] = self.ubx[1]
        mpc.bounds['lower', '_x', 'T_R'] = self.lbx[2]
        mpc.bounds['upper', '_x', 'T_R'] = self.ubx[2]
        mpc.bounds['lower', '_x', 'T_j'] = self.lbx[3]
        mpc.bounds['upper', '_x', 'T_j'] = self.ubx[3]

        # blank tvp
        tvp_template = mpc.get_tvp_template()
        def tvp_fun(t_ind):
            return tvp_template
        mpc.set_tvp_fun(tvp_fun)

        # setup
        mpc.setup()

        # end
        return mpc
