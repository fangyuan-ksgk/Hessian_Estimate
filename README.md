# Hessian_Estimate

Partially Observed Diffusion Process with Discrete Observation (POPDOD) model is a popular model for sequential data inference. It is a special form of Hidden Markov Model (HMM). A singla process X solves a Stochastic differential equation (SDE), and a noisy obseravtion process Y is available at discrete time points. 

This pyton package allows for parameter inference of theta, which models the drift coefficient functions in the SDE, as well as the observation density function. The inference is done through estimation of the Jacobian and Hessian of the log likelihood of the observations with respect to the parameter $\theta$, and then apply gradient-based descent method to obtain value of $\theta$ through training. 

The test file contains a notebook with examples of parameter inference of a special type of PODPDO model.


An example of SGD based on estimated Jacobian to infer the unknown parameter:
![image](https://user-images.githubusercontent.com/66006349/122913820-9c737480-d362-11eb-95db-9f12e3c148de.png)


@ksgk_fangyuan
