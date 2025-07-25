from module import *
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
# plot init
plt.style.use(['science','no-latex'])

# set default device
default_device = 'cpu'

# init system
system = CSTR_dompc(set_seed=0)

## setting seed for repeatability
dm = DataManager(set_seed = 0)

# generate data
dm.random_input_sampler(system = system, n_samples=500)

# set order and split data accordingly
dm.data_splitter(order=1)
#dm.visualize_data()

# train NARX model
dm.train_narx(hidden_layers=[2], batch_size=1000,
          learning_rate=0.1, epochs= 1000, scheduler_flag=True, device=default_device)
#dm.narx.plot_narx_training_history()

# train cqr model
dm.train_cqr(alpha=0.1, hidden_layers=[2], batch_size=1000,
             lr_threshold=1e-7, epochs=1000, scheduler_flag=True, device=default_device)
#dm.cqr.plot_qr_training_history()

# verify qr performance
#dm.cqr_plot_qr_error()

# verify cqr performance
#dm.plot_cqr_error_plotly()

# checking simulator performance
C_a0 = 0
C_b0 = 0
T_R0 = 387.05
T_J0 = 387.05
x_init = np.array([[C_a0, C_b0, T_R0, T_J0]])
#dm.check_simulator(system=system, iter= 50, x_init=x_init)

# check closed loop performance for an MPC with a surrogate model, simulated on the real system
#dm.check_simulator_mpc(system=system, iter=50, setpoint=None, n_horizon= 10, r= 0.01, x_init=x_init)

# run the icb_mpc
R = np.array([[1, 0],
              [0, 1]])
Q = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
iter = 10
setpoint = None
n_horizon = 10
r = 0.1
tightner = 0.1
confidence_cutoff = 0.8
rnd_samples = 7
max_search = 3
dm.case_study_1(system=system, iter=iter, setpoint=setpoint,
                  n_horizon=n_horizon, r=r,
                  tightner=tightner, confidence_cutoff=confidence_cutoff,
                  rnd_samples=rnd_samples, max_search=max_search, R=R, Q=Q, store_gif=True)

# plot generation
dm.plot_simulation(system=system)
dm.show_gif_matplotlib(system = system, gif_name="matplotlib_animation_cs1_main.gif")


#dm.setup_case_study_2(hidden_layers=[10, 10], system=system, setpoint=setpoint,
#                      n_horizon=n_horizon, r=r, epochs=1000, batch_size=1000)
#dm.case_study_2(system=system, iter = iter, x_init=x_init)
#dm.plot_simulation(system=system)

# case study 4
#dm.case_study_5(system=system, iter=10, setpoint=None,
#                  n_horizon=10, r=0.01,
#                  tightner=0.1, confidence_cutoff=0.8, rnd_samples=7, max_search=5,
#                  x_init = x_init, store_gif=True)

#dm.plot_simulation()
#dm.show_gif_matplotlib(system = system, gif_name="matplotlib_animation_cs5_main.gif")

