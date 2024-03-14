import numpy as np
import random

class Dataset:

    def __init__(self, data=None, shuffle=False):
        self.data = data
        self.unique_data = [np.array(a) for a in set([tuple(d) for d in data])]
        if data is not None:
            self.num_data = len(data)
            self.num_unique_data = len(self.unique_data)
            if shuffle:
                random.shuffle(self.data)
        else:
            self.num_data = 0
            self.num_unique_data = 0


    def __iter__(self):
        yield from self.data


    def sample(self):
        return random.choice(self.data)
    

def load_dataset(file_path, data_shape):

    with open(file_path, mode='r') as f:
        lines = f.readlines()
        lines.reverse()
        f.close()

    data_list = []
    data = []
    w, h = data_shape
    while len(lines) > 0:
        line = (lines.pop()).rstrip('\n')
        for c in line:
            if c == '1':
                data.append(1)
            else:
                data.append(-1)
        if len(data) == w*h:
            data_list.append(np.array(data))
            data = []
            if len(lines) > 0:
                lines.pop() # Skip empty line
    return Dataset(data_list, False)
