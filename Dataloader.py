import torch
import numpy as np
import pandas as pd
from torch.utils.data.sampler import WeightedRandomSampler

"""
This file do the data analysis, data clean and data load tasks.

Class DataFrameDataset build a class for load data for training task.
Function load_data load data from 'SFEW.xlsx' and return a data with no name attribute and no nan values.
Function normalization normalize data using z-score method.
"""


# define a customise torch Dataset
class DataFrameDataset(torch.utils.data.Dataset):
    def __init__(self, df, noisy=False):
        self.data_tensor = df
        self.noisy = noisy

    # a function to get items by index
    def __getitem__(self, index):
        input = self.data_tensor[index][1:]
        # Add some random values to input. It is helpful for generalization
        if self.noisy == True:
            input = input * (1 + (torch.rand(1) - 0.5) * 2)  # generalize
        target = self.data_tensor[index][0] - 1
        return input, target

    # a function to count samples
    def __len__(self):
        n, _ = self.data_tensor.shape
        return n


def load_data():
    """
    This function load data from 'SFEW.xlsx' and return a data with no name attribute and no nan values
    :return: Cleaned Data
    """
    data = pd.read_excel(r'SFEW.xlsx', sheet_name=0, header=None)
    data = data.drop(labels=0)  # delete first row
    data = data.drop(labels=0, axis=1)  # delete first column
    data = data._convert(numeric=True)  # convert object to numeric
    data = data.dropna(axis=0)  # delete NAN value (only one)
    return data


def normalization(data):
    """
    This function normalize data using z-score method.
    Other transformation methods have been muted.
    :param data: data
    """
    for i in range(1, 11):
        # data[i] = data[i]-data[i].min()/(data[i].max()-data[i].min())  # [0,1] Norm
        # data[i] = abs(np.log(data[i]))  # log normalize
        data[i] = (data[i] - data[i].mean()) / data[i].std()  # z-score
        # md = np.abs(data[i] - data[i].median())
        # data[i] = (data[i] - data[i].median()) / md.median()


def split_Kdata(data, k, i):
    """
    This function split the data into k sets. Following the method of kfold to split data.
    :param data: data
    :return: train_data, test_data
    """
    data = np.array(data)  # np.ndarray()
    data = data.tolist()
    val = list()
    train = list()
    for idex, item in enumerate(data):
        if idex % k == i:
            val.append(item)
        else:
            train.append(item)
    val = pd.DataFrame(val)
    train = pd.DataFrame(train)
    val[0] = val[0].astype(int)
    train[0] = train[0].astype(int)
    return train, val


def split_data(data):
    """
    This function simply split the data into train set and test set. NOT FOR K-FOLD CROSS VALIDATE.
    About 80 percent training data and 20 percent test data.
    Using this type of spilt makes train set and test set has same distribution of class number.
    :param data: data
    :return: train_data, test_data
    """
    data = np.array(data)  # np.ndarray()
    data = data.tolist()
    train_data = list()
    test_data = list()
    for idex, item in enumerate(data):
        if idex % 10 == 4:
            test_data.append(item)
        else:
            train_data.append(item)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)
    train_data[0] = train_data[0].astype(int)  # keep label value types
    test_data[0] = test_data[0].astype(int)

    return train_data, test_data


# get weight to Balance dataset
def make_weights_for_balanced_classes(data, num_classes):
    """
    Due to the dataset is not much balanced. Number of class 2 and 3 are different with other classes
    This function get a weights for balancing classes.
    :param num_classes: num_classes
    :return: weight
    """
    count = [0] * num_classes
    for i in data[0]:
        count[i - 1] += 1
    weight_per_class = [0.] * num_classes
    N = float(sum(count))
    for i in range(num_classes):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(data)
    for idx, val in enumerate(data[0]):
        weight[idx] = weight_per_class[val - 1]
    return weight
