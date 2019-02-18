import pdb

import numpy as np

from DataPoint import DataPoint


class DataGenerator:
    def __init__(self, distribution):
        self.__distribution = distribution


    def get_data_point(self, label_noise=0.00):
        ind, features, label, feature_costs, label_cost = next(self.__distribution)
        if np.random.rand() < label_noise:
            label = np.random.choice(int(label)+1)
        return DataPoint(ind, features, label, feature_costs, label_cost)
