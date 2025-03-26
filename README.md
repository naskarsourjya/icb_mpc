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

## Closed points

Make an algorithm for random input training.

## Coments from Mentors

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


# Week 3 (27/01/2025)

## Closed Points
The errors between the do_mpc surrogate model and the pytorch narx model are not zero, but very close. It needs to be quantified.
This reduces when order = 1

Checkout splitting of data in narx training and also in make steps. Also check in what order is history for state and input used. (Checked, ref: check excel sheet)

Completely redone data splitter function, where instead of splitting the raw series sequentially, I am generating the ordered narx data adn then randomly splitting data for different purposes.

Start with 100000 for tarining narx, 100000 for cqr calibration, 100 (done)

Query?
Should I keep the current QR training where we train individually each regressor or should we combine the outputs and train all th regressors simultaneously.

What is mod(I2)?? Page 6 of reghu paper 6


# Week 3 (03/02/2025)

## query
1. do we do the calculations for only one miss coverage value(alpha). That means only one pair 
2. np.quantile(arr, 0.7) computes the value below which 70% of the data falls. 
If the 0.7th quantile lies between two data points, linear interpolation is used by default.
3. 
# Week 4 (10/02/2025)
Problems with random MPC whenre the random setpoin tracking is not working

# Week 5 (17/02/2025) (No meeting)
Major resuructuring of one file to multiple classes for modularity. Developed initial version of branching algo.



# Week 6 (24/02/2025)
Rewrote the branching algo in matrices. Massive improvement observed (5hrs+ to couple minutes <10mins).

# Week 7 (03/03/2025)

make plot of simulation with the surrogate model

Plot the setpoints for mpc

use a good madel

make a confidence cutoff for branching algo

Print out an optimal output for the mpc

plot the x_taj form the mpc for verification

# Week 8 (10/03/2025)

## done

Scale data (standard scaling or min max scaling) before training.

add mps support

rewrite calculate surrogate error to calculate errors in matrix instead of make_step

# Week 11 (31/03/2025)

There is translation problem when the pytorch model is converted to do_mpc model. SInce do_mpc is based on casadi which only supports float64, and we are using multiple datatypes in the pytorch, we are getting issues whrn the nn has large nodes or it is very deep.



## todo

Increase the f_ext cost term to 10^3 and then maybe try to reduce if mpc does nothing.

mpc.reset_history is causing problems, find a workaround

Simulate the real system and the surrogate and the cqr, compare all three against each other
(results are out and we are cooked)

Tune system constants to make it fast. k, c and m needs to be tuned.

techniques to prevent overfitting
1. regularisation
2. dropout layers
3. early stopping, stop when loss keeps increasing and restore the best weights
4. reduce parameters

## open points
Change mpc form random set-point tracking to step jump, close to boundary.

Use surrogate mpc to control the real system

Use generic mpc to control real system

Move cqr _post_processing to Pytorch device

Compare results of surrogate mpc and generic mpc

Remove dependence on numpy. Instaed use pytorch.

Look up soruce no 12 from reghu paper


# List of hyperparameters
1. number of layers for narx
2. number of nodes for narx
3. activation function for narx
4. change_probability of random_input_sampler
5. input penalty (r) for random_state_sampler
6. n_horizon for random_state_sampler