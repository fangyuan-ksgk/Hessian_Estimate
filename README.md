# Hessian_Estimate

Partially Observed Diffusion Process with Discrete Observation (POPDOD) model is a popular model for sequential data inference. It is a special form of Hidden Markov Model (HMM). A singla process X solves a Stochastic differential equation (SDE), and a noisy obseravtion process Y is available at discrete time points. 

$$
dX_{t} = a_{\theta}(X_{t})dt + \sigma dW_{t}, X_{0} \sim \eta_{0}
$$
$$
\mathbb{P}(Y_{t} \in dy |X_{t} = x) = g_{\theta}(y,x) dy
$$
where $X_{t} \in \mathbb{R}^{d_x}$, $Y_{t} \in \mathbb{R}^{d_y}$, $W_{t}$ is $d_{x}$ dimensional standrad Brownian motion.

This pyton package allows for parameter inference of $\theta$, which models the drift coefficient functions $a_{\theta}(x)$, as well as the observation density function $g_{\theta}(y,x)$. The inference is done through estimation of the Jacobian and Hessian of the log likelihood of the observations $Y_{1},Y_{2},...,Y_{T}$ with respect to the parameter $\theta$, and then apply gradient-based descent method to obtain value of $\theta$ through training. 

The test file contains a notebook with examples of parameter inference of a special type of PODPDO model.
![equation](http://latex.codecogs.com/gif.latex?O_t%3D%5Ctext%20%7B%20Onset%20event%20at%20time%20bin%20%7D%20t)
![equation](http://latex.codecogs.com/gif.latex?s%3D%5Ctext%20%7B%20sensor%20reading%20%7D) 
![equation](http://latex.codecogs.com/gif.latex?P%28s%20%7C%20O_t%20%29%3D%5Ctext%20%7B%20Probability%20of%20a%20sensor%20reading%20value%20when%20sleep%20onset%20is%20observed%20at%20a%20time%20bin%20%7D%20t)

![image](https://user-images.githubusercontent.com/66006349/122913113-ba8ca500-d361-11eb-9b8e-4233b3385174.png)
