import matplotlib.pyplot as plt
import numpy as np
from omar_utils.percentage_bar import PercentageBar

if __name__ == '__main__':
    from data import DataSet, DataGenerator
    from useful import *
else:
    from .data import DataSet, DataGenerator
    from .useful import *


class LearningAlgorithm:
    """general template for learning algorithms"""
    def __init__(self):
        self.model = None
        self.data = None
        self.converged = None
        self.epoch = None
        self.gradient = None
        self.message = None
        self.l2_regularization = 0.
        self.max_epoch = 1000
        self.max_loss = 1/100
        self.loss = None
        self.test_loss = None
        self.loss_record = []
        self.check_loss_period = 0
        self._keep_training = None
        self.old_loss_weight = .2
        self._bar = PercentageBar(40)

    def compile(self, model=None, data=None):
        """set model and data"""
        if model is not None:
            self.model = model
        if data is not None:
            self.data = data

    def settings(self, max_epoch=None, max_loss=None, l2_reg=None, check_loss_period=None, old_loss_weight=None):
        """
        :param max_epoch: max number of training epoch
        :param max_loss: if loss is less than this training stops
        :param check_loss_period: check loss only once in a while
        :param l2_reg: l2 regularization on weights
        :param old_loss_weight: to avoid terminating training because of unlucky batches, loss is averaged with old loss
        :return: 
        """
        if max_epoch is not None:
            self.max_epoch = max_epoch
        if max_loss is not None:
            self.max_loss = max_loss
        if l2_reg is not None:
            self.l2_regularization = l2_reg
        if check_loss_period is not None:
            self.check_loss_period = check_loss_period
        if old_loss_weight is not None:
            self.old_loss_weight = old_loss_weight

    def compute_loss_dataset(self, data: DataSet):
        """compute average loss of the model on data"""
        assert type(data) == DataSet

        # MSE
        out = 0
        for d in data:
            x = d['x']
            y = d['y']
            net_y = self.model(x)
            if len(y) != len(net_y):
                raise IndexError('Model and dataset must have same output length!')
            for i in range(len(net_y)):
                out += (y[i] - net_y[i]) ** 2 / 2
        out /= len(data)

        # add regularization
        if self.l2_regularization:
            out += sum([i**2 for i in self.model.get_weights()]) * self.l2_regularization / 2

        return out

    def compute_loss(self, data=None):
        """compute average loss of the model on data"""
        if data is None:
            data = self.data
        if type(data) == DataGenerator:
            data = next(data)   # dataset
        return self.compute_loss_dataset(data)

    def update_loss(self, new_loss):
        """append x to the list of loss history"""
        if self.loss is None:
            self.loss = new_loss
        else:
            self.loss = (self.loss * self.old_loss_weight + new_loss) / (self.old_loss_weight + 1)
        self.loss_record += [self.loss]

    def get_loss(self):
        return self.loss

    def test(self, test_data, show=False, dec=2):
        """compute loss on test_data"""
        self.test_loss = self.compute_loss(test_data)

        if show:
            for data in test_data:
                x = '\t'.join([f'{i:.{dec}f}' for i in data['x']])
                y_net = self.model(data['x'])
                y_exp = data['y']
                delta = sum([(y_net[i] - y_exp[i])**2 for i in range(len(y_net))])**.5
                y_net = '\t'.join([f'{i:.{dec}f}' for i in y_net])
                y_exp = '\t'.join([f'{i:.{dec}f}' for i in y_exp])
                print(f"{x} \t\t{y_net} \t\t{y_exp} \t\t{delta:.{dec}f}")

        return self.test_loss

    def get_results(self):
        """return dit describing the results of the training"""
        return {'model': self.model,
                'converged': self.converged,
                'epoch': self.epoch,
                'loss': round(self.loss, 8) if self.loss is not None else None,
                'test loss': round(self.test_loss, 8) if self.test_loss is not None else None,
                'test / max loss': round(self.test_loss / self.max_loss, 2) if self.test_loss is not None else None,
                'message': self.message,
                }

    def plot_loss(self, show=True, title=None, ylim=2., **kwargs):
        """plot the loss over epochs"""
        if len(self.loss_record) <= 1:
            print('Not enough data has been collected for the plot')
        x = [i * self.check_loss_period for i in range(len(self.loss_record))]
        scale = self.loss_record[0]

        # plot loss
        y = np.array(self.loss_record) / scale
        plt.plot(x, y, **kwargs)

        plt.ylim([0, ylim])
        plt.xlabel(title if title else '#batches')
        plt.ylabel(title if title else 'training Loss vs initial loss')
        if show:
            plt.show()

    def check_epochs(self, max_epoch):
        """stop by exceeding number of steps (by setting _keep_training to False)"""
        self.epoch += 1
        self._bar((self.epoch-1)/(max_epoch-1))
        if self.epoch >= max_epoch:
            self._keep_training = False
            self.message = 'Max number of steps has been reached'

    def check_loss(self, max_loss):
        """stop by low loss (by setting _keep_training to False)"""
        if self.check_loss_period == 0:
            compute_loss = False
        else:
            compute_loss = (self.epoch+1) % self.check_loss_period == 0
            if self.epoch == 0:
                compute_loss = True

        if compute_loss:
            self.update_loss(self.compute_loss())
            if self.get_loss() <= max_loss:
                self.converged = True
                self._keep_training = False
                self.message = 'Low-enough loss has been reached'

    def reset_learning(self):
        """reset to initial parameters"""
        self.epoch = 0
        self.converged = False
        self._keep_training = True
        self.message = '-'
        self._bar.reset()

    def add_to_gradient(self, delta_grad):
        """add delta_grad to gradient"""
        if self.gradient is None or self.gradient == []:
            self.gradient = delta_grad
        else:
            self.gradient += delta_grad

    def reset_gradient(self):
        self.gradient = np.array([0. for _ in self.model.get_weights()])

    def compute_gradient_data_point(self, data_point):
        """compute gradient of the model's weights using data_point"""
        x = data_point['x']
        y = data_point['y']
        self.model(x)   # forward propagation
        return self.model.get_gradient(y, l2_reg=self.l2_regularization)

    def compute_gradient_batch(self, batch):
        """compute gradient of the model's weights using batch of data-points"""
        self.check_loss(self.max_loss)
        self.check_epochs(self.max_epoch)
        self.reset_gradient()
        for data_point in batch:
            new_grad = self.compute_gradient_data_point(data_point)
            self.add_to_gradient(new_grad)
        return self.gradient / len(batch)

    def iter_through_data(self):
        """generator of DataSet"""
        if type(self.data) == DataSet:
            while self._keep_training:
                yield self.data
        elif type(self.data) == DataGenerator:
            for batch in self.data:
                if not self._keep_training:
                    break
                if type(batch) == DataSet:
                    yield batch
                else:
                    yield DataSet([batch])

    def learning_algorithm(self, **kwargs):
        """ here you update the weights of the model

        while self._keep_training:
            compute new_weights
            self.model.update_weights(new_weights)
        """
        raise NotImplementedError

    def fit(self, **kwargs):
        """here is where the training happens"""
        self.reset_learning()
        self.learning_algorithm(**kwargs)


class GradientDescent(LearningAlgorithm):
    """
    - compute gradient
    - move the weights slightly in the opposite direction
    """
    def learning_algorithm(self, eta=.1):
        for batch in self.iter_through_data():
            delta = self.compute_gradient_batch(batch) * (-eta)
            self.model.add_to_weights(delta)


class GradientDescent2(LearningAlgorithm):
    """
    - compute gradient
    - move the weights slightly in the opposite direction
    - step size is scaled so that the average length of step size is step_size
    """
    def learning_algorithm(self, step_size=.1):
        avg_step = 0
        n = 0
        for batch in self.iter_through_data():
            n += 1
            delta = self.compute_gradient_batch(batch)
            s = sum([i ** 2 for i in delta]) ** .5
            avg_step = (avg_step * (n-1) + s) / n
            delta *= - step_size / avg_step
            self.model.add_to_weights(delta)


class Momentum(LearningAlgorithm):
    """
    new delta is linear combination of old delta and new gradient
    """
    def learning_algorithm(self, alpha=.9, eta=.1):
        delta = np.array([0. for _ in self.model.get_weights()])
        for batch in self.iter_through_data():
            new_grad = self.compute_gradient_batch(batch)
            delta = delta * alpha - new_grad * eta
            self.model.add_to_weights(delta)


class RProp(LearningAlgorithm):
    """
    - compute gradient
    - move the weights slightly in the opposite direction
    """
    def __init__(self):
        super().__init__()
        self.gradient = None
        self.old_gradient = None
        self.speed = None
        self._old_w = None

    def learning_algorithm(self, max_epoch=100, max_loss=1/100, k_plus=1.2, k_minus=.5, speed0=1/20):
        self._old_w = self.model.get_weights()
        self.speed = [speed0 for _ in self._old_w]
        self.old_gradient = [0. for _ in self._old_w]

        for batch in self.iter_through_data():

            # compute gradient
            self.gradient = self.compute_gradient_batch(batch)

            for i in range(len(self._old_w)):

                product = self.old_gradient[i] * self.gradient[i]
                if product > 0:
                    self.speed[i] *= k_plus
                    self._old_w[i] -= sign(self.gradient[i]) * self.speed[i]
                    self.old_gradient[i] = self.gradient[i]
                elif product < 0:
                    self.speed[i] *= k_minus
                    self.old_gradient[i] = 0
                else:
                    self._old_w[i] -= sign(self.gradient[i]) * self.speed[i]
                    self.old_gradient[i] = self.gradient[i]

            self.model.update_weights(self._old_w)


class NAG(LearningAlgorithm):
    """
    similar to momentum, but gradient is evaluated in:
    old_w  + delta * alpha
    """
    def learning_algorithm(self, alpha=.9, eta=.2):

        delta = np.array([0. for _ in self.model.get_weights()])
        for batch in self.iter_through_data():
            old_w = self.model.get_weights()
            test_w = old_w + delta * alpha
            self.model.update_weights(test_w)
            new_grad = self.compute_gradient_batch(batch)
            delta = delta * alpha - new_grad * eta
            self.model.update_weights(old_w + delta)


class ADAGrad(LearningAlgorithm):
    """
    keep track of how strong is the gradient usually for every dimension
    """
    def __init__(self):
        super().__init__()
        self.G = None

    def learning_algorithm(self, eta=.1, epsilon=10**-8):
        self.G = np.zeros(len(self.model.get_weights()))
        for batch in self.iter_through_data():
            g = self.compute_gradient_batch(batch)
            self.G += [i**2 for i in g]
            delta = g * (-eta)
            delta = [delta[i] / (self.G[i] + epsilon) ** .5 for i in range(len(delta))]
            self.model.add_to_weights(delta)


class RMSProp(LearningAlgorithm):
    """
    like ADAGrad but recent g**2 are more important
    """
    def __init__(self):
        super().__init__()
        self.E = None

    def learning_algorithm(self, eta=.1, gamma=.9, epsilon=10**-8):
        self.E = np.zeros(len(self.model.get_weights()))
        for batch in self.iter_through_data():
            g = self.compute_gradient_batch(batch)
            self.E = [self.E[i] * gamma + g[i] ** 2 * (1 - gamma) for i in range(len(g))]
            delta = g * (-eta)
            delta = [delta[i] / (self.E[i] + epsilon) ** .5 for i in range(len(delta))]
            self.model.add_to_weights(delta)


class AdaDelta(LearningAlgorithm):
    """
    like ADAGrad but doesn't need a learning rate...
    """
    def __init__(self):
        super().__init__()
        self.E_g = None
        self.E_d = None

    def learning_algorithm(self, gamma=.9, epsilon=10**-8):
        self.E_g = np.zeros(len(self.model.get_weights()))
        self.E_d = np.zeros(len(self.model.get_weights()))

        for batch in self.iter_through_data():
            g = self.compute_gradient_batch(batch)
            self.E_g = [self.E_g[i] * gamma + g[i] ** 2 * (1 - gamma) for i in range(len(g))]
            delta = [-g[i] * ((self.E_d[i] + epsilon) / (self.E_d[i] + epsilon)) ** .5 for i in range(len(g))]
            self.E_d = [self.E_d[i] * gamma + delta[i] ** 2 * (1 - gamma) for i in range(len(g))]
            self.model.add_to_weights(delta)
