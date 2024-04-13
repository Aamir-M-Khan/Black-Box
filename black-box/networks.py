from abc import ABC, abstractmethod

class Layer(ABC):
    @abstractmethod
    def forward(self, input):
        pass

    @abstractmethod
    def backward(self, gradient):
        pass

    @abstractmethod
    def parameters(self):
        return
    
    @abstractmethod
    def gradient(self): return