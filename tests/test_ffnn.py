from ffnn import *
from datasets import *
from training import *
import matplotlib.pyplot as plt
from activation_functions import *
from useful import i_print
from random import randrange
from numpy import linspace
from time import time


def test_neuron():
    neuron = Neuron([2, -1], prelu())
    print(neuron)
    print(neuron([-1]))
    print(neuron.output)


def test_layer():
    shape = (2, 3, 2, 1)
    layers = [Layer(length=shape[i+1],
                    input_size=shape[i],
                    act_fun=sigmoid(),
                    weight_range=.1)
              for i in range(len(shape)-1)]

    i_print(layers)

    v = [1, 2]
    for i in range(len(shape)-1):
        v = layers[i](v)
        i_print(v)


def test_nn():
    net = FeedForwardNeuralNet(shape=[2, 2, 1], act_fun=sigmoid())
    net.set_weights([2, -5, -5, -5, 4, 4, 3, -5, -5])
    for d in xor_dataset():
        x = d['x']
        y = d['y']
        out = net(x)[0]
        print(f'{out:.2f} \t {y}')


def test_gradient():
    net = FeedForwardNeuralNet(shape=[2, 2, 1], act_fun=sigmoid())
    net.set_weights([2, -5, -5, -5, 4, 4, 3, -5, -5])
    net([0, 0])
    for i in net.get_gradient([1]):
        print(f'{i:+.3f}')


def test_training(weight_range=.1, length=2, noise=.05, batch_size=20, max_epoch=1000, max_loss=.01, step_size=.1):
    data_train = noisy_xor(noise).batch(batch_size)
    data_test = xor_dataset()

    net = FeedForwardNeuralNet(shape=[length, 4, 1], act_fun=[prelu(), sigmoid()], weight_range=weight_range)

    algorithm = GradientDescent2()
    algorithm.compile(model=net, data=data_train)

    algorithm.settings(max_epoch=max_epoch // 2, max_loss=max_loss*5)
    algorithm.fit(step_size=step_size*5)

    algorithm.settings(max_epoch=max_epoch // 2, max_loss=max_loss)
    algorithm.fit(step_size=step_size)

    algorithm.test(data_test)
    d_print(algorithm.get_results())
    print(f'\nLoss = {algorithm.get_results()["loss"]:.3f}')

    algorithm.plot_loss(ylim=[0, .25])


def test_training_2(max_epoch=4000, batch_size=15, max_loss=1/100):
    data_gen = mod_dataset()
    train_data = data_gen.batch(batch_size)
    test_data = data_gen.to_dataset(200)

    net = FeedForwardNeuralNet(shape=[4, 2, 1], act_fun=[sigmoid(), sigmoid()], weight_range=.1)

    algorithm = AdaDelta()
    algorithm.compile(model=net, data=train_data)
    algorithm.settings(max_epoch=max_epoch, max_loss=max_loss, check_loss_period=10, l2_reg=10**-4, old_loss_weight=1.)
    algorithm.fit()

    algorithm.test(test_data)

    d_print(algorithm.get_results())
    algorithm.plot_loss()


def test_training_3(n=10):
    data = xor_dataset()

    for _ in range(n):
        net = FeedForwardNeuralNet(shape=[2, 2, 1], act_fun=tanh(), weight_range=.5)
        algorithm = GradientDescent()
        algorithm.compile(model=net, data=data)
        algorithm.fit(max_epoch=300, max_loss=1 / 50, eta=.1)
        algorithm.plot_loss(show=False, ylim=[0, 1], alpha=.25, color='b')

    plt.show()


def test_save():
    data = xor_dataset()
    net = FeedForwardNeuralNet(shape=[2, 2, 1], act_fun=tanh(), weight_range=.4)
    net.settings(name='saves\\test_01')

    algorithm = GradientDescent()
    algorithm.compile(model=net, data=data)

    algorithm.fit(max_epoch=300, max_loss=1 / 50, eta=.2)

    net.save()

    for i in net.get_weights():
        print(f'{i:+.3f}')
    print()


def test_load():
    net = FeedForwardNeuralNet(shape=[2, 2, 1], act_fun=tanh(), weight_range=.4)
    net.settings(name='saves\\test_01')
    net.load()

    for i in net.get_weights():
        print(f'{i:+.3f}')

    print()
    print(net([1, 1]))


def test_new(k=5):
    data = xor_dataset()
    net = FeedForwardNeuralNet(shape=[2, 2, 1], act_fun=sigmoid())
    algorithm = GradientDescent()
    algorithm.compile(model=net, data=data)

    best = 1
    while True:
        w = [randrange(-k, 1+k) for _ in range(9)]
        w[2] = w[1]
        w[2+3] = w[1+3]
        w[2+6] = w[1+6]
        net.set_weights(w)
        new_loss = algorithm.compute_loss()
        if new_loss < best:
            best = new_loss
            print(f'{best:.4f} \t{w}')


def test_battery(max_epoch=4000, max_loss=1/200, n=20, batch_size=10, tolerance=.5, k0=1., blocks=3, k_scale=10):

    data_train = mod_dataset().batch(batch_size)
    data_test = mod_dataset().to_dataset(100)

    record = []
    t = time()

    print('test loss / max loss')

    for _ in range(n):

        # build model
        net = FeedForwardNeuralNet(shape=[4, 2, 1],
                                   act_fun=[sigmoid(), sigmoid()],
                                   weight_range=2.)

        # init algorithm
        algorithm = ADAGrad()
        algorithm.compile(model=net, data=data_train)
        algorithm.settings(check_loss_period=5, l2_reg=10 ** -6)

        # learning
        for j in range(blocks):
            algorithm.settings(max_epoch=(max_epoch * (j+1))//blocks, max_loss=max_loss)
            algorithm.fit(eta=k0/k_scale**j)

        # test
        algorithm.test(data_test)
        results = algorithm.get_results()

        test_loss = results["loss"]
        converged = test_loss <= max_loss * (1 + tolerance)

        print(f'{test_loss / max_loss:.2f}', end='\t')

        if converged:
            record += [results['epoch']]
            print('converged')
        else:
            print()

        algorithm.plot_loss(show=False, color='blue' if converged else 'red', alpha=.5)

    print(f'\nconvergence rate: {len(record)/n*100:.1f}%')
    if len(record):
        print(f'avg steps when converges: {sum(record) / len(record):.0f}')
    print(f't = {time() - t:.1f} s')
    plt.show()


def test_momentum(max_epoch=1000):
    data = xor_dataset()

    print(data)

    net = FeedForwardNeuralNet(shape=[2, 2, 1], act_fun=[sigmoid(), sigmoid()], weight_range=1.)

    algorithm = Momentum()
    algorithm.compile(net, data)
    algorithm.settings(max_epoch=max_epoch, max_loss=1/100, l2_reg=10**-5, check_loss_period=10)
    # algorithm.fit(alpha=.9, eta=.1)
    algorithm.fit()
    algorithm.test(data)
    d_print(algorithm.get_results())

    print()
    for d in data:
        x = d['x']
        y = d['y']
        print(x, y, f'{net(x)[0]:.2f}')

    algorithm.plot_loss(ylim=[0, .16])


def test_rprop(max_epoch=200):
    data = xor_dataset()
    net = FeedForwardNeuralNet(shape=[2, 2, 1], act_fun=[sigmoid(), sigmoid()], weight_range=1 / 20)
    algorithm = RProp()
    algorithm.compile(net, data)
    algorithm.settings(max_epoch=max_epoch, max_loss=1/100, l2_reg=10**-5)
    algorithm.fit(k_plus=1.2, k_minus=.5, speed0=1/20)
    algorithm.test(data)
    d_print(algorithm.get_results())
    algorithm.plot_loss(ylim=[0, .16])


def test_2(max_steps=100, max_loss=1/100, n=20):
    data_train = xor_dataset()
    data_test = data_train

    for par in linspace(1., 2., 11):

        count = 0

        for _ in range(n):
            net = FeedForwardNeuralNet(shape=[2, 3, 1],
                                       act_fun=[sigmoid(), sigmoid()],
                                       weight_range=.2)

            algorithm = RProp()
            algorithm.compile(model=net, data=data_train)
            algorithm.settings(max_epoch=max_steps, max_loss=max_loss, check_loss_period=10, l2_reg=10**-5)
            algorithm.fit(k_plus=par)
            algorithm.test(data_test)

            results = algorithm.get_results()
            converged = results['converged']
            count += converged

        plt.scatter(par, count / n, color='b')

    plt.show()


if __name__ == '__main__':
    from random import seed
    seed(1)

    # test_neuron()
    # test_layer()
    # test_nn()
    # test_gradient()
    # test_training()
    # test_training_3()
    # test_save()
    # test_load()
    # test_new()
    # test_momentum()
    # test_rprop()
    # test_2()
    # test_training_2()
    test_battery()
