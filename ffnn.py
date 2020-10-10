import pickle
from numpy import array, concatenate

if __name__ == "__main__":
    from model import Model
    from useful import *
else:
    from .model import Model
    from .useful import *


class Neuron:

    def __init__(self, weights, act_fun):
        self.weights = array(weights)
        self.act_fun = act_fun
        self.output = None

    def __getitem__(self, item):
        return self.weights[item]

    def __setitem__(self, key, value):
        self.weights[key] = value

    def __len__(self):
        return len(self.weights)

    def __call__(self, x):
        return self.compute_output(x)

    def __repr__(self):
        w = ', '.join([f'{i:+.2f}' for i in self.weights])
        return f'Neuron({w})'

    def compute_output(self, x):
        activation = self.weights.dot(concatenate(([1], x)))
        self.output = self.act_fun(activation)
        return self.output


class Layer:

    def __init__(self, length, input_size, act_fun, weight_range=1/20):
        self.length = length
        self.input_size = input_size
        self.act_fun = act_fun
        self.neurons = None
        self.init_neurons(weight_range)

    def __getitem__(self, item):
        return self.neurons[item]

    def __len__(self):
        return self.length

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def __repr__(self):
        return f'Layer({self.length})'

    def init_neurons(self, weight_range):
        self.neurons = [Neuron(random_weights(self.input_size + 1, weight_range), self.act_fun)
                        for _ in range(len(self))]


class FeedForwardNeuralNet(Model):

    def __init__(self, shape: list, act_fun, weight_range=.1):
        super().__init__()
        self.shape = shape
        self.act_fun = act_fun
        self.layers = None
        self.input = None
        self.output = None
        if type(act_fun) == list:
            assert len(act_fun) == len(shape) - 1
        self.init_net(weight_range)

    def __getitem__(self, item):
        return self.layers[item]

    def __iter__(self):
        """yield indices of weights"""
        return self.iter_weight_indices()

    def __call__(self, x):
        """compute output"""
        self.input = x
        self.output = x
        for layer in self.layers:
            self.output = layer(self.output)
        return self.output

    def __repr__(self):
        return f'FFNN({self.shape})'

    def iter_weight_indices(self):
        """these are the indexes you have to iter over in order to iterate over the weights"""
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i])):
                for k in range(len(self.layers[i][j])):
                    yield (i, j, k)

    def iter_output_indices(self):
        """these are the indexes you have to iter over in order to iterate over the outputs"""
        for i in range(len(self.layers) + 1):
            for j in range(len(self.layers[i])):
                yield (i, j)

    def get_act_fun(self, i_layer):
        """get activation function of the i_layer"""
        if type(self.act_fun) == list:
            return self.act_fun[i_layer]
        else:
            return self.act_fun

    def get_quick_d(self, i_layer):
        """
        the quick derivative computes the derivative of the activation function
        of the i_layer but using directly the outputs
        """
        return self.get_act_fun(i_layer).quick_d

    def init_net(self, weight_range):
        """
        build the layers of the net
        init weights with random values
        """
        self.layers = [Layer(length=self.shape[i+1],
                             input_size=self.shape[i],
                             act_fun=self.get_act_fun(i),
                             weight_range=weight_range)
                       for i in range(len(self.shape)-1)]

    def set_weights(self, w):
        """add the weights to w (plain list)"""
        n = 0
        for i, j, k in self.iter_weight_indices():
            if w[n] is not None:
                self[i][j][k] = w[n]
            n += 1

    def get_weight(self, i_layer, j_destination, k_origin):
        """get weight w_ijk"""
        return self[i_layer][j_destination][k_origin]

    def get_weights(self):
        """get all weights as plain list"""
        return array([self.get_weight(*i) for i in self.iter_weight_indices()])

    def get_y(self, i_layer, j_neuron):
        """get output y_ij of neuron i, j"""
        return self[i_layer-1][j_neuron].output if i_layer else self.input[j_neuron]

    def update_weights(self, new_weights):
        """add w to the weights (plain list)"""
        for n, indices in enumerate(self):
            i, j, k = indices
            self[i][j][k] = new_weights[n]

    def add_to_weights(self, new_weights):  # todo compulsory?
        """add w to the weights (plain list)"""
        for n, indices in enumerate(self):
            i, j, k = indices
            self[i][j][k] += new_weights[n]

    def _iter_neuron_indices(self):
        """yield tuples of indices defining each neuron"""
        for i in reversed(range(len(self.shape) - 1)):
            for j in range(self.shape[i]):
                yield i, j

    def _iter_weight_indices(self):
        for i in range(len(self.shape) - 1):
            for j in range(self.shape[i + 1]):
                for k in range(self.shape[i] + 1):
                    yield i, j, k

    def get_gradient_y(self, expect):
        """compute gradient for activations"""
        y = [[0. for _ in range(i)] for i in self.shape]
        y[-1] = self.output - array(expect)
        for i, j in self._iter_neuron_indices():
            y[i][j] = sum([y[i+1][n] * self.get_weight(i, n, j + 1) * self.get_quick_d(i)(self.get_y(i + 1, n))
                           for n in range(self.shape[i+1])])
        return y

    def get_gradient(self, expect: list, l2_reg=0.):
        """compute gradient of cost in respect to each weight"""
        y = self.get_gradient_y(expect)
        w = [self.get_quick_d(i)(self.get_y(i+1, j)) * (self.get_y(i, k-1) if k else 1) * y[i+1][j]
             for i, j, k in self._iter_weight_indices()]
        if l2_reg:
            w0 = self.get_weights()
            w += w0 * l2_reg
        return w

    # ----- file handling -----

    def save(self, path=None):
        """save weights from pickle file"""
        if path:
            with open(path, 'wb') as file:
                pickle.dump(self.get_weights(), file)
        else:
            with open(self.name + '.pkl', 'wb') as file:
                pickle.dump(self.get_weights(), file)

    def load(self, path=None):
        """load weights from pickle file (return True if successful else false)"""
        if path:
            with open(path, 'rb') as file:
                self.set_weights(pickle.load(file))
        else:
            with open(self.name + '.pkl', 'rb') as file:
                self.set_weights(pickle.load(file))

