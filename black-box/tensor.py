import numpy as np

class Tensor:
    def __init__(self, value):
        self.value = np.array(value)

    def __str__(self):
        return str(self.value.__str__())