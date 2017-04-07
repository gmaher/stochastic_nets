import numpy as np

class DataGenerator(object):
    def __init__(self, num_samples, noise, x_dim):
        self.num_samples = num_samples
        self.noise = noise
        self.x_dim = x_dim  # can be 2

    def genGauss(self, mean):
        data = []
        for _ in range(self.num_samples):
            # x is 2 dimensional
            features = np.random.normal(mean, self.noise, size=self.x_dim).tolist()
            data.append(features)

        return data

    def generate_two_gaussian(self):
        x_data1 = self.genGauss(self.noise)  # 2
        x_data2 = self.genGauss(-self.noise) # -2

        x_data = []
        x_data.extend(x_data1); x_data.extend(x_data2)

        # x_train = x_data[0:self.split, :]
        # x_test = x_data[self.split:, :]

        # second half is 1
        y_data = np.array([1]*self.num_samples * 2, dtype=np.int32)
        y_data[:self.num_samples] = 0  # first half is 0

        # y_train = y_data[:self.split]
        # y_test = y_data[self.split:]

        # use get_batch method
        return np.array(x_data, dtype=np.float32), y_data