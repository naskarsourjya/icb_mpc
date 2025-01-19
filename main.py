from module import *

# init
spring_system = SpringSystem()
su = SurrogateCreator()

# pipeline to generate data and store
su.random_state_sampler(system = spring_system, n_samples=50)
#su.random_input_sampler(system = spring_system, n_samples=1000)
su.data_test_train_splitter(train_split=0.3)
# su.visualize2d(system=spring_system)
#ann.store_raw_data(filename='data\spring_random_1000.pkl')

# pipeline to load and visualize data
#ann.load_raw_data(filename='data\\spring_random_1000.pkl')
#ann.visualize()

su.narx_trainer(order=2, hidden_layers=[10], batch_size=32,
          learning_rate=0.001, epochs= 1000)
#ann.save_narx(filename='data\\narx10_10_s1000_o1.pkl')

#ann.load_narx(filename='data\\narx10_10_s1000_o1.pkl')
#print(su.narx_make_step(states=np.array([[0.1, 0.2, 0.3],
#                               [0.4, 0.5, 0.6]]),
#              inputs=np.array([[0.7, 0.8, 0.9]])))

#su.narx_2_dompc()
#ann.save_surrogate(filename='data\\surrogate1.pkl')

#su.simulator_set_initial_guess(states=np.array([[0.1, 0.2, 0.3],
#                               [0.4, 0.5, 0.6]]),
#              inputs=np.array([[0.7, 0.8]]))

#print(su.simulator_make_step(u0=np.array([[0.9]])))

print(su.surrogate_error(system=spring_system))