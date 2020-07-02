import torch
import numpy as np
import torch.nn.functional as F
# This file contains some fucntions for helping work and testing

# define a function to plot confusion matrix
def plot_confusion(input_sample, num_classes, des_output, actual_output):
    confusion = torch.zeros(num_classes, num_classes)
    for i in range(input_sample):
        actual_class = actual_output[i]
        predicted_class = des_output[i]

        confusion[actual_class][predicted_class] += 1

    return confusion


def count_nueron(net):
    """
    This function count sum of the number of hidden neurons and inputs
    :param net: checking neural network
    :return: sum of the number of hidden neurons and inputs
    """
    count = 0
    for name, param in net.named_parameters():
        if 'weight' in name:
            count += len(param[0])
    return count


# def weight_decay(epoch, num_neuron, k):
#     """
#     This function calculate the number of weight decay for current epoch.
#     This funciton refer to N.K. Treadgold and T.D. Gedeon 1997
#     :param epoch: current training epoch
#     :param num_neuron: number of hidden neurons
#     :param k: a user defined parameter which effects the magnitude of weight decay used
#     :return: difference the weight should decay
#     """
#     param = np.log(k * epoch * num_neuron)*100
#     decay = 1 / param
#     return decay
# print(weight_decay(10, 5, 500))


def get_parameter_number(net):
    # Get parameter numbers in neural netwrok.
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# rewrite torch.max() funcitom
def torch_max(output):
    predicted_label = list()
    softmax_output = F.softmax(output, 1)
    for record in softmax_output:
        max_value = max(record)
        if max_value < 0.9: # Set threshold for classification
            predicted_label.append(-1)
        else:
            predicted_label.append(1)
    return predicted_label

def revers_torch_max(output):
    # This funciton helps to calculate FN,TN
    predicted_label = list()
    softmax_output = F.softmax(output, 1)
    for record in softmax_output:
        max_value = max(record)
        if max_value >= 0.98:
            predicted_label.append(-1)
        else:
            predicted_label.append(1)
    return predicted_label
