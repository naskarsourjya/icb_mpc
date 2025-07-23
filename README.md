# Iterative Constricting Boundary Nominal Model Predictive Control
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

Scale input data (standard scaling or min max scaling) before training. Done.

add mps support. Done.

rewrite calculate surrogate error to calculate errors in matrix instead of make_step. Done.

# Week 11 (31/03/2025)

There is translation problem when the pytorch model is converted to do_mpc model. SInce do_mpc is based on casadi which only supports float64, and we are using multiple datatypes in the pytorch, we are getting issues when the nn has large nodes or it is very deep.

Tune system constants to make it fast. k, c and m needs to be tuned.

# Week 12 (07/04/2025)

mpc.reset_history is causing problems, find a workaround

Simulate the real system and the surrogate and the cqr, compare all three against each other
(results are out and we are cooked)

# Week 13 (14/04/2025)

Ensure when confidence = 1, cqr does not branch out and algo is equivalent to nominal mpc. Done.

Reduce constraints only for states which are violated. Done.

Discuss mpc cloning problem (tvp initialisation issue): Sol: re-init mpc in every make step. Have a new set-point state instead of tvp. Done

# Week 14 (21/04/2025)

adding random points in between (monte carlo style). Done.

If the bounds are too outside, the lbx may go above ubx. Should we have an upper limit for the adjustment factor? Fixed. We are throwing out an assertion error stating that the tightner might need some reduction.

I am thinking about multiplying probability values with the states, because generally the states with the lower probability values are the ones which go further away from the boundary. Done.

# Week 15 (30/04/2025)

Issue: The surrogate model and the real model simulation profiles do not match, not even remotely. Fixed. Issue with the NARXing of the data.

# Week 16 (07/05/2025)

Collect data nmpc vs robust mpc. Hyperparameter tuning.

Make midterm ppt.

Check trial input timestamps and shape, should connect with the previous input.

rewrite this, all lag values should be -1:

state_1_lag_1 <<--- @1=((state_1_lag_1--0.000122589)/0.00352367), @2=((state_2_lag_1--0.000468091)/0.0228845), @3=((state_1_lag_2--9.18993e-05)/0.00353787), @4=((state_2_lag_2-7.97144e-05)/0.0226143), @5=((input_1_lag_0--0.00115011)/0.0581787), @6=((input_1_lag_1--0.0021914)/0.0580589), @7=(((((((0.0421667*@1)+(0.0105288*@2))+(0.0642783*@3))+(0.0207141*@4))+(-0.129506*@5))+(0.041167*@6))+-0.00440907), @8=(((((((-0.0127768*@1)+(0.0332566*@2))+(-0.00804659*@3))+(-0.00634837*@4))+(0.156069*@5))+(-0.0296747*@6))+-0.0277647), (((0.0307126*((exp(@7)-exp((-@7)))/(exp(@7)+exp((-@7)))))+(0.035189*((exp(@8)-exp((-@8)))/(exp(@8)+exp((-@8))))))+0.000948778)
state_2_lag_1 <<--- @1=((state_1_lag_1--0.000122589)/0.00352367), @2=((state_2_lag_1--0.000468091)/0.0228845), @3=((state_1_lag_2--9.18993e-05)/0.00353787), @4=((state_2_lag_2-7.97144e-05)/0.0226143), @5=((input_1_lag_0--0.00115011)/0.0581787), @6=((input_1_lag_1--0.0021914)/0.0580589), @7=(((((((0.0421667*@1)+(0.0105288*@2))+(0.0642783*@3))+(0.0207141*@4))+(-0.129506*@5))+(0.041167*@6))+-0.00440907), @8=(((((((-0.0127768*@1)+(0.0332566*@2))+(-0.00804659*@3))+(-0.00634837*@4))+(0.156069*@5))+(-0.0296747*@6))+-0.0277647), (((-0.0795851*((exp(@7)-exp((-@7)))/(exp(@7)+exp((-@7)))))+(0.0739332*((exp(@8)-exp((-@8)))/(exp(@8)+exp((-@8))))))+0.0016903)
state_1_lag_2 <<--- state_1_lag_1
state_2_lag_2 <<--- state_2_lag_1
input_1_lag_1 <<--- input_1_lag_0

# Week 17 (14/05/2025)

Inside the icbmpc algo, if the states cross boundary further away (but boundaries not exceeded in the next state), the boundaries are still constricted. Should we have a cutoff for this, to reduce iterations?

# Week 20 (05/06/2025)

1. remove dependence of cqr probability on per layer. Done
2. remove the dependence of the first point. Done
3. do scaling for y as well. Train with scaled values then unscale. Done
4. implement robust mpc with LQR gain (linearize around nominal state only)
5. fix issue with Klatt Engell CSTR example

# Week 21

1. check simulator
2. check cqr coverage with the teset data

discuss:
1. pytorch to do-mpc translation. Is scaling needed, since the translation already has the scaler and unscaler implemented in the model itself. Done.
2. Penalise both inputs for the cstr. the function step state mpc cannot be a generic function. It needs to be specific to the example. Pull it out to main.py instead of _datamanager.py. Done.

# Week 22:
1. reduce the time stamp, and reduce the simulation  time to 20%.
2. Fix plot. D0ne
3. Greedy cost function (maximze C_b). Done
4. Investigate parameters Q and R for lqr effect on solution.
3. Implement the the midterm icb mpc algo
4. Write Theory outline and sections in overleaf.
5. Implement the multi stage mpc example
6. Implement Klatt Engell CSTR



## todo

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

# pressentation
introcude the problem with bad case study
introduce the case study, explaing the spring damper system very well
intorduce approach to deal with bad surrogate models and introduce the tuning knobs, maybe a block diagram

compare many models, hyperparameters
iba_mpc with poor model and improvement
next steps, impelement parallelisation, give runtime numbers

# Thesis story line
1. Abstract
2. Introduction: Motivation, take from presentation
3. Theory: MPC, NN, NARX, CQR, LQR; Path integral control
4. Method: ICB MPC
5. Results: Case Study, Discussion
6. Conclusion

# Thesis Notes
 1. Add stuff about the history and inception of MPC for chapter 2 where MPC is discussed in detail.
 2. My initial thought was that Robust MPC with the nominal, cqr high, cqr low models would work better because in Robust MPC no input shifting policy is explicitly chosen, apart from the non-anticipatory constraints. But now that I think about it, my algo is a bit better because the not only we have the upper and lower bounds of the cqr model, but we also have random points between the upper and lower limits, helping us out in case of systems which are not monotonic.
 3. add a case study for a highly non monotonic system.
