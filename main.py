from module import *
import numpy as np

# init
default_device = 'cpu'
spring_system = SpringSystem(set_seed=0)
dm = DataManager(set_seed = 0)

# pipeline to generate data and store
#su.random_state_sampler(system = spring_system, n_samples=50)
dm.random_input_sampler(system = spring_system, n_samples=10000)
dm.data_splitter(order=3)
#dm.visualize2d_data()
#ann.store_raw_data(filename='data\spring_random_1000.pkl')

# pipeline to load and visualize data
#ann.load_raw_data(filename='data\\spring_random_1000.pkl')
#dm.visualize_data()

#su.narx_trainer(hidden_layers=[10], batch_size=320,
#          learning_rate=0.1, epochs= 100)
dm.narx_trainer(hidden_layers=[2], batch_size=1000,
          learning_rate=0.1, epochs= 1000, scheduler_flag=True, device=default_device)
#su.plot_narx_training_history()
#ann.save_narx(filename='data\\narx10_10_s1000_o1.pkl')

#ann.load_narx(filename='data\\narx10_10_s1000_o1.pkl')
#print(su.narx_make_step(states=np.array([[0.1, 0.2, 0.3],
#                                         [0.4, 0.5, 0.6]]),
#              inputs=np.array([[0.7, 0.8, 0.9]])))

#su.narx_2_dompc()
#ann.save_surrogate(filename='data\\surrogate1.pkl')

#su.simulator_set_initial_guess(states=np.array([[0.1, 0.2, 0.3],
#                               [0.4, 0.5, 0.6]]),
#              inputs=np.array([[0.7, 0.8]]))

#print(su.simulator_make_step(u0=np.array([[0.9]])))

dm.train_individual_qr(alpha=0.1, hidden_layers=[2], batch_size=1000,
             lr_threshold=1e-7, epochs=1000, scheduler_flag=True, device=default_device)
#su.train_individual_qr(alpha=0.2, hidden_layers=[10], batch_size=320)
#su.cqr.plot_qr_training_history_plotly()
#dm.cqr_plot_qr_error()
#dm.plot_cqr_error_plotly()
#su.cqr_set_initial_guess(states=np.array([[0.1, 0.2, 0.3],
#                               [0.4, 0.5, 0.6]]),
#              inputs=np.array([[0.7, 0.8]]))

#print(su.cqr_make_step(u0=np.array([[0.9]])))
#su.plot_cqr_error_plotly()
#dm.run_simulation(system=spring_system, iter=2, n_horizon=10, r=0.01, store_gif=True)
#dm.cqr_mpc.plot_trials()
#dm.plot_simulation()
#dm.show_gif()

dm.check_simulator(system=spring_system, iter= 50)
#dm.run_simulation(system=spring_system, iter=10, n_horizon=10, r=0.01, tightner=1,
#                  confidence_cutoff=0.8, rnd_samples=7, setpoint=0.0099, max_search=5, store_gif=True)
#dm.plot_simulation()
#dm.show_gif_matplotlib()


