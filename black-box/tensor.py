import numpy as np

class Tensor:   
    def __init__(self, data, requires_grad=False, creators=[], creation_op=None, id=None):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data)       
        self.creators = creators
        self.creation_op = creation_op

    def backward(self):
        topo_order = []
        visited = set()
        
        def topo_sort(v):
            if v not in visited:
                visited.add(v)
                for parent in v.creators:
                    topo_sort(parent)
                topo_order.append(v)
        
        topo_sort(self)
        self.grad = np.ones_like(self.data)
        
        for v in reversed(topo_order):
            if v.creation_op:
                v.creation_op.backward(v)
   
    def __add__(self, other):
        return Add()(self, ensure_variable(other))

    def __neg__(self):
        return Neg()(self)
    
    def __sub__(self, other):
        return Sub()(self, ensure_variable(other))
    
    def __mul__(self, other):
        return Mul()(self, ensure_variable(other))

    def __pow__(self, other): 
        return Pow(other)(self)

    def __truediv__(self, other):
        return Div()(self, ensure_variable(other))

    def mean(self, axis=None):
        return Mean(axis)(self)
    
    # def sum(self, dim):
    #     return Sum.forward(self, dim)
    
    # def expand(self, dim, copies):
    #     return Expand.forward(self, dim, copies)
    
    # def transpose(self):
    #     return Transpose.forward(self)
    
    # def mm(self, x):
    #     return MM.forward(self, x)
    
    def __repr__(self):
        return str(self.data.__repr__())
    
    def __str__(self):
        return str(self.data.__str__())

def ensure_variable(x):
    return x if isinstance(x, Tensor) else Tensor(x)

class Operation:
    def __call__(self, *inputs):
        self.inputs = inputs
        requires_grad = any(input.requires_grad for input in inputs)
        self.output = Tensor(data=self.forward(*[input.data for input in inputs]), requires_grad=requires_grad)
        self.output.creation_op = self
        self.output.creators = inputs
        return self.output

    def forward(self, *inputs):
        raise NotImplementedError
    
    def backward(self, output):
        raise NotImplementedError
    
class Add(Operation):
    @staticmethod
    def forward(a, b):
        return a + b

    def backward(self, output):
        a, b = self.inputs
        a.grad += output.grad
        b.grad += output.grad


class Sub(Operation):
    @staticmethod
    def forward(a, b):
        return a - b

    def backward(self, output):
        a, b = self.inputs
        a.grad += output.grad
        b.grad -= output.grad

class Mul(Operation):
    @staticmethod
    def forward(a, b):
        return a * b

    def backward(self, output):
        a, b = self.inputs
        a.grad += b.data * output.grad
        b.grad += a.data * output.grad

class Div(Operation):
    @staticmethod
    def forward(a, b):
        return a / b

    def backward(self, output):
        a, b = self.inputs
        a.grad += output.grad / b.data
        b.grad += (-a.data * output.grad)/b.data**2

class Pow(Operation):
    def __init__(self, power):
        self.power = power
    
    def forward(self, a):
        return a ** self.power
    
    def backward(self, output):
        a, = self.inputs
        a.grad += self.power * (a.data ** (self.power - 1)) * output.grad

class ConstPow(Operation):
    def __init__(self, power):
        self.power = power
    
    def forward(self, a):
        return a ** self.power
    
    def backward(self, output):
        a, = self.inputs
        a.grad += self.power * (a.data ** (self.power - 1)) * output.grad

class Mean(Operation):
    def __init__(self, axis=None):
        self.axis = axis

    def forward(self, a):
        return np.mean(a, axis=self.axis)
    
    def backward(self, output):
        a = self.inputs
        grad = output.grad / np.prod(a.shape)
        a.grad += np.broadcast_to(grad, a.data.shape)

class Clip(Operation):
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, a):
        self.mask = (a >= self.min_val) & (a <= self.max_val)
        return np.clip(a, self.min_val, self.max_val)
    
    def backward(self, output):
        a = self.inputs
        a.grad += self.mask * output.grad

class Reshape(Operation):
    def __init__(self, shape) -> None:
        self.shape = shape

    def forward(self, a):
        self.orignal_shape = a.shape
        return np.reshape(a, self.shape)

    def backward(self, output):
        a = self.inputs
        a.grad += np.reshape(output.grad, self.orignal_shape)

class Squeeze(Operation):
    def __init__(self, axis=None):
        self.axis = axis

    def forward(self, a): 
        self.orignal_shape = a.shape
        return np.squeeze(a, axis=self.axis) 
    
    def backward(self, output):
        a = self.inputs
        a.grad += np.reshape(output.grad, self.orignal_shape)

class Where(Operation):
    def forward(self, condition, x, y):
        return np.where(condition, x, y)

    def backward(self, output):
        condition, x, y = self.inputs
        condition_grad = np.zeros_like(condition.data)

        x_grad = np.where(condition.data, output.grad, 0)
        y_grad = np.where(~condition.data, output.grad, 0)

        x.grad += x_grad
        y.grad += y_grad

# class Neg(Operation):
#     @staticmethod
#     def forward(a):
#         return -a

#     def backward(self, output):
#         a = self.inputs
#         a.grad = -1 * output.grad
        

# class Sum:
#     @staticmethod
#     def forward(tensor, dim):
#         requires_grad = tensor.requires_grad
#         creators = [tensor]
#         data = tensor.data.sum(dim)
#         return Tensor(data, requires_grad, creators, f"sum_{dim}")

#     @staticmethod
#     def backward(grad, tensor):
#         dim = int(tensor.creation_op.split("_")[1])
#         tensor.creators[0].backward(grad.expand(dim, tensor.creators[0].data.shape[dim]))


# class Expand:
#     @staticmethod
#     def forward(tensor, dim, copies):
#         requires_grad = tensor.requires_grad
#         creators = [tensor]
#         trans_cmd = list(range(0, len(tensor.data.shape)))
#         trans_cmd.insert(dim, len(tensor.data.shape))
#         new_data = tensor.data.repeat(copies).reshape(list(tensor.data.shape) + [copies]).transpose(trans_cmd)
#         return Tensor(new_data, requires_grad, creators, f"expand_{dim}")

#     @staticmethod
#     def backward(grad, tensor):
#         tensor.creators[0].backward(grad.sum(tensor.data.ndim - 1))


# class Transpose:
#     @staticmethod
#     def forward(tensor):
#         requires_grad = tensor.requires_grad
#         creators = [tensor]
#         data = tensor.data.transpose()
#         return Tensor(data, requires_grad, creators, "transpose")

#     @staticmethod
#     def backward(grad, tensor):
#         tensor.creators[0].backward(grad.transpose())


# class MM:
#     @staticmethod
#     def forward(a, b):
#         requires_grad = a.requires_grad or b.requires_grad
#         creators = [a, b]
#         data = a.data.dot(b.data)
#         return Tensor(data, requires_grad, creators, "mm")

#     @staticmethod
#     def backward(grad, tensor):
#         a, b = tensor.creators
#         grad_a = grad.data.dot(b.data.transpose())
#         grad_b = grad.data.transpose().dot(a.data)
#         a.backward(Tensor(grad_a))
#         b.backward(Tensor(grad_b))


# class Neg:
#     @staticmethod
#     def forward(tensor):
#         requires_grad = tensor.requires_grad
#         creators = [tensor]
#         data = tensor.data * -1
#         return Tensor(data, requires_grad, creators, "neg")

#     @staticmethod
#     def backward(grad, tensor):
#         tensor.creators[0].backward(Tensor(grad.data * -1))