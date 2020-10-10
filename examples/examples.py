from ffnn import FeedForwardNeuralNet
from activation_functions import sigmoid
from training import NAG


def example(game, n=5):
    """
    train neural network

    :param game: val_fun -> list of data of game [{'x':[...], 'y': [...]} ... ]
    :return:
    """
    shape = [2, 4, 1]
    act_fun = sigmoid()
    fit_kwargs = {}
    max_epoch = 10 ** 3

    model = FeedForwardNeuralNet(shape=shape, act_fun=act_fun)
    model.settings(name='model')
    model.load()

    algorithm = NAG()
    algorithm.compile(model=model)
    algorithm.settings(max_epoch=max_epoch, check_loss_freq=0)

    for _ in range(n):
        training_data = game(val_fun=model)
        algorithm.compile(data=training_data)
        algorithm.fit(**fit_kwargs)
        model.save()


if __name__ == '__main__':
    example()
