from module import *
import numpy as np

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
dm.visualize_data()

# train NARX model
dm.train_narx(hidden_layers=[2], batch_size=1000,
          learning_rate=0.1, epochs= 1000, scheduler_flag=True, device=default_device)
dm.narx.plot_narx_training_history()

# train cqr model
dm.train_cqr(alpha=0.1, hidden_layers=[2], batch_size=1000,
             lr_threshold=1e-7, epochs=1000, scheduler_flag=True, device=default_device)
dm.cqr.plot_qr_training_history()

# verify qr performance
dm.cqr_plot_qr_error()

# verify cqr performance
dm.plot_cqr_error_plotly()

# checking simulator performance
C_a0 = 0
C_b0 = 0
T_R0 = 387.05
T_J0 = 387.05
x_init = np.array([[C_a0, C_b0, T_R0, T_J0]])
dm.check_simulator(system=system, iter= 50, x_init=x_init)

# check closed loop performance for an MPC with a surrogate model, simulated on the real system
dm.check_simulator_mpc(system=system, iter=50, setpoint=1.99, n_horizon= 10, r= 0.01, x_init=x_init)

# run the icb_mpc
R = np.array([[1, 0],
              [0, 1]])
Q = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
dm.check_icbmpc(system=system, iter=10, setpoint=0.0099,
                  n_horizon=10, r=0.01,
                  tightner=0.1, confidence_cutoff=0.8, rnd_samples=7, max_search=5, R=R, Q=Q,
                  x_init = x_init, store_gif=True)

# plot generation
dm.plot_simulation()
dm.show_gif_matplotlib()


