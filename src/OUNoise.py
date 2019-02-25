import numpy as np
import copy
import random

class OUNoise:
    def __init__(self, action_size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(action_size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.action_size = action_size        
        self.reset()

    """
    def reset(self, epsilon):
        self.state = copy.copy(self.mu)
        self.epsilon = epsilon
    """
    def reset(self):
        self.state = copy.copy(self.mu)
    
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        #dx = self.theta * (self.mu - x) + self.epsilon * self.sigma * np.random.randn(self.action_size)
        self.state = x + dx
        return self.state