import numpy as np


class DataPoint():
	def __init__(self, ind, features, label, feature_costs, label_cost):
		self.__features = features
		self.__label = label
		self.__feature_costs = feature_costs
		self.__label_cost = label_cost

		self.__accumulated_cost = 0
		self.__known_features = np.ones(len(features))*np.nan
		self.__is_label_known = False
		self.__index = ind


	def get_features(self):
		return self.__features * self.__known_features


	def get_label(self):
		if self.__is_label_known:
			return self.__label
		else:
			return np.nan


	def request_label(self):
		self.__accumulated_cost += self.__label_cost
		self.__is_label_known = True

		return self.get_label()


	def request_feature(self, index):
		self.__accumulated_cost += self.__feature_costs[index]
		self.__known_features[index] = 1

		return self.get_features()


	def get_accumulated_cost(self):
		return self.__accumulated_cost


	def get_feature_costs(self):
		return self.__feature_costs


	def get_label_costs(self):
		return self.__label_costs
    
	def get_index(self):
		return self.__index
