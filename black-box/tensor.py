import numpy as np

class Tensor:
    def __init__(self, value):
        self.value = np.array(value)

    def __str__(self):
        return str(self.value)
    
    def __repr__(self):
        return str(self.value.__repr__())
    
    def __add__(self, other):
        return Tensor(self.value + other.value)
    
    def __sub__(self, other):
        return Tensor(self.value - other.value)
    
    