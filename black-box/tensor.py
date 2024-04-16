import numpy as np

class Tensor:
    
    stack = []  # Class-level stack to track operations
    
    def __init__(self, data, requires_grad=False, creators=None, creation_op=None, id=None):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        if id is None:
            self.id = np.random.randint(0, 100000)
        else:
            self.id = id
        
        self.creators = creators
        self.creation_op = creation_op
        self.children = {}
        
        if creators is not None:
            for c in creators:
                if self.id not in c.children:
                    c.children[self.id] = 1
                else:
                    c.children[self.id] += 1
        
        # Push this tensor and its creation operation onto the stack
    #
        #Tensor.push_stack()

    @classmethod
    def push_stack(cls):
        # Pushes the current tensor and its creation operation onto the stack
        if hasattr(cls, 'stack'):
            cls.stack.append((id(cls), cls.creation_op))

    @classmethod
    def pop_stack(cls):
        # Pops the top element from the stack
        if hasattr(cls, 'stack') and cls.stack:
            return cls.stack.pop()
        return None

    def all_children_grads_accounted_for(self):
        for id, cnt in self.children.items():
            if cnt != 0:
                return False
        return True 
    
    def backward(self, grad=None, grad_origin=None):
        if self.requires_grad:
            if grad is None:
                grad = Tensor(np.ones_like(self.data))

            if grad_origin is not None:
                if self.children[grad_origin.id] == 0:
                    raise Exception("Cannot backprop more than once")
                else:
                    self.children[grad_origin.id] -= 1

            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad
            
            # grads must not have grads of their own
            assert grad.requires_grad == False
            
            # Only continue backpropagation if there's something to
            # backprop into and if all gradients (from children)
            # are accounted for. Override waiting for children if
            # "backprop" was called on this variable directly
            if self.creators is not None and \
               (self.all_children_grads_accounted_for() or grad_origin is None):

                if self.creation_op == "add":
                    Add.backward(self.grad, self)
                    
                if self.creation_op == "sub":
                    Sub.backward(self.grad, self)

                if self.creation_op == "mul":
                    Mul.backward(self.grad, self)

                if self.creation_op == "mm":
                    MM.backward(self.grad, self)
                    
                if self.creation_op == "transpose":
                    Transpose.backward(self.grad, self)

                if "sum" in self.creation_op:
                    Sum.backward(self.grad, self)
                    
                if "expand" in self.creation_op:
                    Expand.backward(self.grad, self)
                    
                if self.creation_op == "neg":
                    Neg.backward(self.grad, self)
    
    def __add__(self, other):
        return Add.forward(self, other)

    def __neg__(self):
        return Neg.forward(self)
    
    def __sub__(self, other):
        return Sub.forward(self, other)
    
    def __mul__(self, other):
        return Mul.forward(self, other)   

    def sum(self, dim):
        return Sum.forward(self, dim)
    
    def expand(self, dim, copies):
        return Expand.forward(self, dim, copies)
    
    def transpose(self):
        return Transpose.forward(self)
    
    def mm(self, x):
        return MM.forward(self, x)
    
    def __repr__(self):
        return str(self.data.__repr__())
    
    def __str__(self):
        return str(self.data.__str__())


class Add:
    @staticmethod
    def forward(a, b):
        requires_grad = a.requires_grad or b.requires_grad
        creators = [a, b]
        data = a.data + b.data
        return Tensor(data, requires_grad, creators, "add")

    @staticmethod
    def backward(grad, tensor):
        tensor.creators[0].backward(grad)
        tensor.creators[1].backward(grad)


class Sub:
    @staticmethod
    def forward(a, b):
        requires_grad = a.requires_grad or b.requires_grad
        creators = [a, b]
        data = a.data - b.data
        return Tensor(data, requires_grad, creators, "sub")

    @staticmethod
    def backward(grad, tensor):
        tensor.creators[0].backward(grad)
        tensor.creators[1].backward(Tensor(-grad.data))


class Mul:
    @staticmethod
    def forward(a, b):
        requires_grad = a.requires_grad or b.requires_grad
        creators = [a, b]
        data = a.data * b.data
        return Tensor(data, requires_grad, creators, "mul")

    @staticmethod
    def backward(grad, tensor):
        a, b = tensor.creators
        grad_a = grad * b
        grad_b = grad * a
        a.backward(grad_a)
        b.backward(grad_b)


class Sum:
    @staticmethod
    def forward(tensor, dim):
        requires_grad = tensor.requires_grad
        creators = [tensor]
        data = tensor.data.sum(dim)
        return Tensor(data, requires_grad, creators, f"sum_{dim}")

    @staticmethod
    def backward(grad, tensor):
        dim = int(tensor.creation_op.split("_")[1])
        tensor.creators[0].backward(grad.expand(dim, tensor.creators[0].data.shape[dim]))


class Expand:
    @staticmethod
    def forward(tensor, dim, copies):
        requires_grad = tensor.requires_grad
        creators = [tensor]
        trans_cmd = list(range(0, len(tensor.data.shape)))
        trans_cmd.insert(dim, len(tensor.data.shape))
        new_data = tensor.data.repeat(copies).reshape(list(tensor.data.shape) + [copies]).transpose(trans_cmd)
        return Tensor(new_data, requires_grad, creators, f"expand_{dim}")

    @staticmethod
    def backward(grad, tensor):
        tensor.creators[0].backward(grad.sum(tensor.data.ndim - 1))


class Transpose:
    @staticmethod
    def forward(tensor):
        requires_grad = tensor.requires_grad
        creators = [tensor]
        data = tensor.data.transpose()
        return Tensor(data, requires_grad, creators, "transpose")

    @staticmethod
    def backward(grad, tensor):
        tensor.creators[0].backward(grad.transpose())


class MM:
    @staticmethod
    def forward(a, b):
        requires_grad = a.requires_grad or b.requires_grad
        creators = [a, b]
        data = a.data.dot(b.data)
        return Tensor(data, requires_grad, creators, "mm")

    @staticmethod
    def backward(grad, tensor):
        a, b = tensor.creators
        grad_a = grad.data.dot(b.data.transpose())
        grad_b = grad.data.transpose().dot(a.data)
        a.backward(Tensor(grad_a))
        b.backward(Tensor(grad_b))


class Neg:
    @staticmethod
    def forward(tensor):
        requires_grad = tensor.requires_grad
        creators = [tensor]
        data = tensor.data * -1
        return Tensor(data, requires_grad, creators, "neg")

    @staticmethod
    def backward(grad, tensor):
        tensor.creators[0].backward(Tensor(grad.data * -1))