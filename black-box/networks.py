from abc import ABC, abstractmethod
import numpy as np
import tensor as Tensor


class Parameter:
    def __init__(self) -> None:
        pass

class Module:
    """
    A base class for building neural network modules.

    Attributes:
        hook (callable): A hook function to be called after the forward pass.
        params (list): A list of parameters registered in the module.
        children (list): A list of child modules.
        _training (bool): A flag indicating whether the module is in training mode.

    """  
    def __init__(self) -> None:
        """
        Initializes the module with no hhok, 
        an empty list of parameters, 
        and an empty list of children.
        Sets the training mode to False.
        """
        self.hook = None
        self.params = []
        self.children = []
        self._training = False

    def register_parameters(self, *ps):
        """
        Registers parameters to the module.

        Args:
            *ps: Parameters to be registered.
        """
        self.params.extend(ps)

    def register_modules(self, *ms):
        """
        Registers child modules to the module.

        Args:
            *ms: Modules to be registered.
        """
        self.children.extend(ms)

    @property
    def training(self):
        """
        Returns:
            bool: The training state of the modules.
        """
        return self._training
    
    @training.setter
    def training(self, value):
        """
        Sets the training state of the module and propogates it to all child modules.

        Args:
            value (bool): The training state to set.
        """
        self._training = value
        for m in self.children:
            m.training = value

    def parameters(self):
        """
        Returns:
            list: A list of all parameters in the module and its children.
        """
        return self.params + [p for m in self.children for p in m.parameters()]
    
    def __setattr__(self, name: str, value: np.Any) -> None:
        """
        Overrides the default __setattr__ to register parameters and modules automatically.

        Args:
            name (str): The name of the attribute.
            value: The value of the attribute.
        """
        super().__setattr__(name, value)
        if isinstance(value, Parameter):
            self.register_parameters(value)
        if  isinstance(value, Module):
            self.register_modules(value)

    def __call__(self, *args: np.Any, **kwds: np.Any) -> np.Any:
        """
        
        """
        result = self.forward(*args, **kwds)
        if self.hook is not None:
            self.hook(result, args)
        return result
    
    def cuda(self):
        """
        Moves all parameters of the module and its children to the GPU.
        """
        for p in self.parameters():
            p.data = p.data.cuda()

    def forward(self, *args, **kwargs):
        """
        Defines the computation performed at every call.
        Should be overwritten by all subclasses.

        Args:
            *args: Positional arguements.
            **kwargs: Keyword arguements.

        Returns:
            The result of the computation.
        """
        raise NotImplementedError("Forward method not implemented")


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self._layers = layers

    def forward(self, input):
        for layer in self._layers:
            return layer(input)
            

class Linear(Module):
    def __init__(self, input_size, output_size, bias):
        super().__init__()
        self.W = Tensor(np.random.randn(input_size, output_size)/np.sqrt(input_size), requires_grad=True)
        self.bias = Tensor(np.random.randn(output_size)/np.sqrt(input_size))  # TODO:check this
        self.is_bias = bias

    def forward(self, input_feature):
        y = input_feature@self.W
        if self.is_bias:
            y += self.bias
        return y

class Dropout(Module):
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None

    def forward(self, input, training=True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.p, size=input.shape) / (1- self.p)
            return input * self.mask
        else:
            return input

class BatchNorm1d:
    def __init__(self, num_features, epsilon=1e-5, momentum=0.1) -> None:
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum

        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, input, training=True):
        if training:
            input_mean = np.mean(input, axis=0, keepdims=True)
            input_var = np.var(input, axis=0, keepdims=True)

            normalized_input = (input - input_mean) / np.sqrt(input_var + self.epsilon)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * input_mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * input_var.squeeze()

        else:
            normalized_input = (input - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

        return self.gamma * normalized_input + self.beta 

class BatchNorm2d:
    def __init__(self, num_features, epsilon=1e-5, momentum=0.1) -> None:
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum

        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, input, training=True):
        if training:
            input_mean = np.mean(input, axis=(0, 2, 3), keepdims=True)
            input_var = np.var(input, axis=(0, 2, 3), keepdims=True)

            normalized_input = (input - input_mean) / np.sqrt(input_var + self.epsilon)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * input_mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * input_var.squeeze()

        else:
            normalized_input = (input - self.running_mean[:, None, None]) / np.sqrt(self.running_var[:, None, None] + self.epsilon)

        return self.gamma[:, None, None] * normalized_input + self.beta[:, None, None] 
    
class LayerNorm(Module):
    def __init__(self, num_features, epsilon=1e-5):
        self.num_features = num_features
        self.epsilon = epsilon

        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

    def forward(self, input):
        input_mean = np.mean(input, axis=-1, keepdims=True)
        input_var = np.var(input, axis=-1, keepdims=True)

        input_normalized = (input - input_mean) / np.sqrt(input_var + self.epsilon)

        # Scale and Shift
        output = self.gamma * input_normalized + self.var

        return output 

class GroupNorm(Module):
    def __init__(self, num_groups, num_channels, epsilon=1e-5) -> None:
        super().__init__()
        self.num_groups= num_groups
        self.num_channels = num_channels
        self.epsilon = epsilon

        self.gamma = np.ones(num_channels)
        self.beta = np.zeros(num_channels)

    def forward(self, input):
        N, C, H, W = input.shape
        G = self.num_groups
        assert C%G == 0, "num_channels must be divisble by num_groups"

        # Rehape input to (N, G, C//G, H, W)
        input = input.reshape(N, G, C//G, H, W)

        # Compute mean and variance along the (N, G, H, W) dimensions
        input_mean = np.mean(input, axis=(2, 3, 4), keepdims=True)
        input_var = np.var(input, axis=(2, 3, 4), keepdims=True)

        # Normalize the input
        normalized_input = (input - input_mean) / np.sqrt(input_var + self.epsilon)

        # Reshape the input back to (N, C, H, W)
        normalized_input = normalized_input.reshape(N, C, H, W)

        # Scale and Shift
        out = self.gamma[:, None, None] * normalized_input + self.beta[:, None, None]

        return out

#### Activation Functions
class ReLU(Module):
    pass

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass
    


#### Loss Functions
class CrossEntropyLoss(Module):
    pass

class Softmax(Module):
    pass

class MSELoss(Module):
    pass


####


