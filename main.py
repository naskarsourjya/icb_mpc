from module import *

# init
spring_system = SpringSystem(set_seed=0)
su = SurrogateCreator(set_seed = 0)

# pipeline to generate data and store
#su.random_state_sampler(system = spring_system, n_samples=50)
su.random_input_sampler(system = spring_system, n_samples=100)
su.data_splitter(order=3)
#su.visualize2d_data()
#ann.store_raw_data(filename='data\spring_random_1000.pkl')

# pipeline to load and visualize data
#ann.load_raw_data(filename='data\\spring_random_1000.pkl')
#ann.visualize()

#su.narx_trainer(hidden_layers=[10], batch_size=320,
#          learning_rate=0.1, epochs= 100)
su.narx_trainer(hidden_layers=[1], batch_size=320,
          learning_rate=0.1, epochs= 1000, scheduler_flag=True)
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

su.train_individual_qr(alpha=0.2, hidden_layers=[10], batch_size=320,
             train_threshold=1e-7, epochs=1000, scheduler_flag=True)
#su.train_individual_qr(alpha=0.2, hidden_layers=[10], batch_size=320)
#su.plot_qr_training_history_plotly()
#su.plot_qr_error()
su.conform_qr()
#su.cqr_set_initial_guess(states=np.array([[0.1, 0.2, 0.3],
#                               [0.4, 0.5, 0.6]]),
#              inputs=np.array([[0.7, 0.8]]))

#print(su.cqr_make_step(u0=np.array([[0.9]])))
#su.plot_cqr_error_plotly()
su.run_simulation(system=spring_system, iter=50, n_horizon=10, r=0.01)
su.plot_simulation()

