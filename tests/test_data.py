from datasets import *
from useful import i_print


def test_data_set():
    i_print(mod_dataset(module=4, input_len=4), 10)


def test_1():
    dg = DataGenerator(((i, i * 10) for i in range(6)), do_save=True)
    # i_print(dg)
    # i_print(dg.saved_data)

    ds = dg.to_dataset(3)
    print(ds)


def test_2():
    dg = DataSet([(i, 'a'*(i+1)) for i in range(6)])
    i_print(dg)
    dg.shuffle()
    i_print(dg)


def test_3(n=4):
    """loop over data in bunches of n"""
    dg = DataSet([(i, 'a'*(i+1)) for i in range(6)])
    i_print(dg.loop_iter().batch(4), n=n, separator='\n')


def test_4():
    data = xor_dataset()
    i_print(data)
    i_print(data)


def test_5():
    ds = DataSet([{'x': [1, 2], 'y': [1]}])
    print(ds.input_size())
    print(ds.output_size())


if __name__ == '__main__':
    test_data_set()
