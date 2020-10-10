from random import shuffle, randrange

if __name__ == '__main__':
    from useful import *
else:
    from .useful import *


class DataSet:

    def __init__(self, data: list):
        self.data = data
        self._i = 0
        assert hasattr(self.data, '__len__')

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def __next__(self):
        i = self._i
        self._i += 1
        if self._i > len(self):
            self._i = 0
            raise StopIteration
        return self[i]

    def head(self, n=5):
        if len(self) <= n:
            return '\n' + '\n'.join([str(i) for i in self])
        else:
            return '\n' + '\n'.join([str(self[i]) for i in range(n)]) + f'\n... and {len(self) - n} more...\n'

    def __repr__(self):
        return self.head()

    def __getitem__(self, item) -> dict:
        return self.data[item]

    def add_data(self, data_point):
        self.data += [data_point]

    def shuffle(self):
        shuffle(self.data)
        return self

    def pick_random(self):
        return self[randrange(len(self))]

    def loop_iter(self):
        return DataGenerator((self.data[i] for i in loop_gen(len(self))()))

    def batch(self, batch_size):
        return self.loop_iter().batch(batch_size)

    def input_size(self):
        return len(self[0]['x'])

    def output_size(self):
        return len(self[0]['y'])

    def split(self, *args):
        """split data-set into smaller ones, args are sizes in percentage (last size can be omitted)

        Example:
        len(ds) = 100
        ds.split(.5, .3) -> ds[0:50], ds[50:80], ds[0:100]

        """
        partial = [sum(args[:j + 1]) for j in range(len(args))]
        if partial[-1] > 1:
            raise ValueError('Sum of arguments must be less than 1')
        indices = [round(i * len(self)) for i in [0] + partial + [1]]
        sizes = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
        slices = [slice(indices[i], indices[i + 1]) for i in range(len(indices) - 1)]
        return [DataSet(self[slices[i]]) for i in range(len(slices)) if sizes[i]]


class DataGenerator:

    def __init__(self, itr, do_save=False):
        """

        :param itr: yields one data-point at the time
        :param do_save:
        """
        self.itr = itr
        self.do_save = do_save
        self.saved_data = DataSet([])

    def __iter__(self):
        return self

    def __next__(self):
        data = next(self.itr)
        if self.do_save:
            self.saved_data.add_data(data)
        return data

    def batch(self, batch_size):
        """returns the next data as DataSet"""
        def gen():
            local_data = []
            for data in self:
                local_data += [data]
                if len(local_data) == batch_size:
                    yield DataSet(local_data)
                    local_data = []
        return DataGenerator(gen())

    def get_next(self, n):
        """"""
        return gen_next_n(n)(self)

    def to_dataset(self, length):
        return DataSet(list(self.get_next(length)))


def weighted_random_data_gen(dataset: DataSet):
    """"""
    cumulative = [0.]
    sum_ = 0
    for d in dataset:
        if 'weight' not in d:
            raise KeyError('data-points must have "weight" key')
        if d['weight'] < 0:
            raise ValueError("weight can't be negative")
        sum_ += d['weight']
        cumulative += [sum_]
    if sum_ == 0:
        raise ValueError('weights should not add up tp 0')

    def itr():
        while True:
            r = random() * sum_
            i_bot = 0
            i_top = len(cumulative) - 1
            while i_top - i_bot > 1:
                i_mid = (i_bot + i_top) // 2

                if cumulative[i_mid] < r:
                    i_bot = i_mid
                else:
                    i_top = i_mid
            yield dataset[i_top-1]

    return DataGenerator(itr())
