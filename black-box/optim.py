import numpy as np


class SGD:
    def __init__(self, parameters, lr=1e-3):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for parameter in self.parameters:
            parameter.data -= parameter.grad.data * self.alpha

    def zero(self):
        for parameter in self.parameters:
            parameter.grad.data *= 0
        


class Adam:
    def __init__(self,):
        pass

    def step(self):
        pass

    def zero(self):
        pass

class Adadelta:
    def __init__(self,): 
        pass

    def step(self):
        pass

    def zero(self):
        pass

