# Hessian_Estimate

Partially Observed Diffusion Process with Discrete Observation (POPDOD) model is a popular model for sequential data inference. It is a special form of Hidden Markov Model (HMM). A singla process X solves a Stochastic differential equation (SDE), and a noisy obseravtion process Y is available at discrete time points. 

This pyton package allows for parameter inference of $\theta$, which models the drift coefficient functions in the SDE, as well as the observation density function. The inference is done through estimation of the Jacobian and Hessian of the log likelihood of the observations $Y_{1},Y_{2},...,Y_{T}$ with respect to the parameter $\theta$, and then apply gradient-based descent method to obtain value of $\theta$ through training. 

The test file contains a notebook with examples of parameter inference of a special type of PODPDO model.


![image.png](attachment:a72407f4-2509-47c6-bc4b-b3210d4876ca.png)
