import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd.functional import hessian
from torch.autograd.functional import jacobian as jac
import progressbar
tod = torch.distributions
from tensorflow.keras.utils import Progbar
import pkbar
from torch.distributions import Categorical

def main():
    print('running')
