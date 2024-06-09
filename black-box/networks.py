from abc import ABC, abstractmethod
import numpy as np
import tensor as T


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
            

class Linear(Layer):
    pass


#### Activation Functions
class ReLU(Layer):
    pass

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass
    


#### Loss Functions
class CrossEntropyLoss(Layer):
    pass

class Softmax(Layer):
    pass

class MSELoss(Layer):
    pass



