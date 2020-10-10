
class Model:

    def __init__(self):
        self.length = None
        self.name = 'model'

    def get_gradient(self, **kwargs):
        """return gradient of the loss in respect to weights once the output has been computed"""
        raise NotImplementedError

    def update_weights(self, new_weights: list):
        """overwrite the current weights with new_weights"""
        raise NotImplementedError

    def get_weights(self):
        """get all weights as plain list"""
        raise NotImplementedError

    def __len__(self):
        if self.length is None:
            self.length = len(self.get_weights())
        return self.length

    def settings(self, name):
        self.name = name
