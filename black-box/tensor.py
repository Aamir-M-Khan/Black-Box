import numpy as np

class Tensor:
    def __init__(self, value, operator, requires_grad):
        self.value = np.array(value)
        self.creators = []
        self.operator = operator
        self.requires_grad = requires_grad
        self.children = {}

        if id is None:
            id = np.random.randint(0, 1,00,000)
        self.id = id

        if self.creators is not None: 
            for c in self.creators:
                if self.id not in c.children:
                    c.children[self.id] = 1
                else: 
                    c.children[self.id] += 1

    def all_children_grads_accounted_for(self): 
        for id, cnt in self.children.items():
            if cnt != 0: 
                return False 
        return True
    
    def __str__(self):
        return str(self.value)
    
    def __repr__(self):
        return str(self.value.__repr__())
    
    def __add__(self, other):
       # return Add.forward(self, other)
        if self.requires_grad & other.requires_grad:
            return Tensor(self.value + other.value,
                        creators=[self, other],
                        operator = "add",
                        requires_grad = True)
        return Tensor(self.value + other.value)
        
    def __sub__(self, other):
        if self.requires_grad & other.requires_grad:
            return Tensor(self.value - other.value,
                        creators=[self, other],
                        operator = "sub",
                        requires_grad = True)            
        return Tensor(self.value - other.value)
    
    def __mul__(self, other):
        if self.autograd & other.autograd:
            return Tensor(self.value * other.value,
                        creators=[self, other],
                        operator = "mul",
                        autograd = True)
        return Tensor(self.value * other.value)
    
    def backward(self, grad=None, grad_origin=None):
        # if not self.requires_grad:
        #     return
        
        # if gradient is None:
        #     gradient = Tensor(np.ones_like(self.data))

        # self.gradient = gradient

        # operations = [self]

        # while operations:
        #     op = operations.pop()
        #     if op.gradient is None:
        #         continue
        #     op.backward()
        #     operations.extend(op.parents)

        if self.requires_grad: 
            if grad_origin is not None:
                if self.children[grad_origin.id] == 0:
                    raise Exception("cannot backpropogate more than once")
                else:
                    self.children[grad_origin.id] -= 1

                if self.grad is None:
                    self.grad = grad 
                else:
                    self.grad += grad 

                if self.creators is not None and self.all_children_grads_accounted_for() or grad_origin is None:
                    if self.creation_op == "add":
                        self.creators[0].backward(self.grad, self)
                        self.creators[1].backward(self.grad, self)




