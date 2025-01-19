# nmpc_cqr_gpu_thesis
Hey, I am Sourjya Naskar, and this is my Master's Thesis, during my Masters in Process Systems Engineering at TU Dortmund, 2025. This code executes CQR on Nvidia GPU to guarantee the results of an MPC.

# Christmas holiday
when i put the following boundaries in spring system
        self.t_step = 0.1
        self.n_horizon = 10
        self.r = 0.25   # penalty for input
                        # range betn [0 -1]
                        # increasing this reduces oscillatroy behaviou of input
        # box constraints
        self.lbx = np.array([-1, -1])   # [lower_bound_position, lower_bound_velocity]
        self.ubx = np.array([1, 10])     # [upper_bound_position, upper_bound_velocity]
        self.lbu = np.array([-1])       # [lower_bound_f_ext]
        self.ubu = np.array([1])        # [upper_bound_f_ext]

the system goes out of the box constraints. Thus, I put up a RuntimeError in the main loop to stop execution if data generation goes cukoo!


# Week 1 (13/01/2025)

Queries from Mentors

Why random trajectory mpc? This is makin git incresingly complex, especiallly the presence of an mpc making it substantially slower.

I chose this in the first place to sensure the samples stay within the box constraints. In case the initial condition is very close to the boundary of the system, then there might exist inputs which push it outside the box constraints.

But in our case since we intend to tighten up constraints which might be violated anyway some information beyond the box constrints might be useful. So I have now decide to go for one random initial state and simulate system based on random feasible inputs. Also pray to god that the system stays inside the constriants.

How to use narx model in do mpc framework use all history of inputs and all states along with history as states.

Add a validation for the do_mpc vs the pytorch narx model.

The errors are not zero, but very close. It needs to be quantified.

# Week 2 (20/01/2025)

1. Performance better on random state trajectory method (smaller errors with same no of samples). Also, in case of random input trajectory, the states far exceed the boundaries.
2. What are the inputs for the CQR models, same as NARX?
3. Confirm test train split!! train-->NARX, calibration-->CQR and test split-->validation and performance metrics.
4. 

Queries from Mentors

# open points
1. The errors between the do_mpc surrogate model and the pytorch narx model are not zero, but very close. It needs to be quantified.
2. Make an algorithm for random input training.
3. Checkout splitting of data in narx training and also in make steps. Also check in what order is history for state and input used. (Checked, ref: check excel sheet)

# List of hyperparameters
1. number of layers for narx
2. number of nodes for narx
3. activation function for narx
4. change_probability of random_input_sampler
5. input penalty (r) for random_state_sampler
6. n_horizon for random_state_sampler