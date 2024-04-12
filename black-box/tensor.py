import numpy as np

class Tensor:
    def __init__(self, value, creators, operator, autograd):
        self.value = np.array(value)
        self.creators = creators
        self.operator = operator
        self.autograd = autograd


    def __str__(self):
        return str(self.value)
    
    def __repr__(self):
        return str(self.value.__repr__())
    
    def __add__(self, other):
        if self.autograd & other.autograd:
            return Tensor(self.value + other.value,
                        creators=[self, other],
                        operator = "add",
                        autograd = True)
        return Tensor(self.value + other.value)
        
    def __sub__(self, other):
        if self.autograd & other.autograd:
            return Tensor(self.value - other.value,
                        creators=[self, other],
                        operator = "sub",
                        autograd = True)            
        return Tensor(self.value - other.value)
    
    def __mul__(self, other):
        if self.autograd & other.autograd:
            return Tensor(self.value * other.value,
                        creators=[self, other],
                        operator = "mul",
                        autograd = True)
        return Tensor(self.value * other.value)
    
    def backward(self, gradient=None):
        if not self.requires_grad:
            return
        
        if gradient is None:
            gradient = Tensor(np.ones_like(self.data))

        self.gradient = gradient

        operations = [self]

        while operations:
            op = operations.pop()
            if op.gradient is None:
                continue
            op.backward()
            operations.extend(op.parents)


