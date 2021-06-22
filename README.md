# Hessian_Estimate

Partially Observed Diffusion Process with Discrete Observation (POPDOD) model is a popular model for sequential data inference. It is a special form of Hidden Markov Model (HMM). A singla process X solves a Stochastic differential equation (SDE), and a noisy obseravtion process Y is available at discrete time points. 
- <img src="https://latex.codecogs.com/gif.latex?O_t=\text { Onset event at time bin } t " /> 

$$
dX_{t} = a_{\theta}(X_{t})dt + \sigma dW_{t}, X_{0} \sim \eta_{0}
$$
$$
\mathbb{P}(Y_{t} \in dy |X_{t} = x) = g_{\theta}(y,x) dy
$$
where $X_{t} \in \mathbb{R}^{d_x}$, $Y_{t} \in \mathbb{R}^{d_y}$, $W_{t}$ is $d_{x}$ dimensional standrad Brownian motion.

This pyton package allows for parameter inference of $\theta$, which models the drift coefficient functions $a_{\theta}(x)$, as well as the observation density function $g_{\theta}(y,x)$. The inference is done through estimation of the Jacobian and Hessian of the log likelihood of the observations $Y_{1},Y_{2},...,Y_{T}$ with respect to the parameter $\theta$, and then apply gradient-based descent method to obtain value of $\theta$ through training. 

The test file contains a notebook with examples of parameter inference of a special type of PODPDO model.
