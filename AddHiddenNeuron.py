import torch
import torch.nn as nn


input_size = 10
num_classes = 7

def addNeuron(net):
    """
    When this function been called, the Casper will be added a new Hidden Neuron.
    This funciton takes the advantages of flexible of Pytorch. ModuleDict() is used to transfer out built layers
    to original network.
    :param net: the training network
    :return: optimizer for new training network which has added a new Hidden Neuron
    """
    net.n_hidden_layers += 1
    net.input_hidden_layers[str(len(net.input_hidden_layers))] = nn.Linear(input_size, 1, bias=False)
    for n_connection in range(net.n_hidden_layers - 1):
        net.hidden_hidden_layers[str(len(net.hidden_hidden_layers))] = nn.Linear(1, 1, bias=False)  # add bias for new neurons
    net.hidden_output_layers[str(len(net.hidden_output_layers))] = nn.Linear(1, num_classes, bias=True)
    '''
    Set different learning rate to different layers!
    - Region L1: Weights connec/ng to new neuron. 
    – Region L2: Weights connected from new neuron to output neurons. 
    – Region L3: Remaining weights (all weights connected to and coming from the old hidden and input neurons).
    L1>>L2>L3 
    The value of L1, L2 and L3 are 0.2, 0.005 and 0.001. Refer to the technique paper
    '''
    L1_params = list(map(id, net.input_hidden_layers[str(len(net.input_hidden_layers)-1)].parameters()))  # L1
    L2_params = list()
    L2_params+= list(map(id, net.hidden_output_layers[str(len(net.hidden_output_layers)-1)].parameters()))  # L2
    for num in range(net.n_hidden_layers - 1):
        L2_params += list(map(id, net.hidden_hidden_layers[str(len(net.hidden_hidden_layers)-num-1)].parameters()))  # L2
    L1L2 = L1_params + L2_params

    base_params = filter(lambda p: id(p) not in L1L2,
                         net.parameters())  # L3
    params = [
        {'params': base_params, 'lr': 0.001},  # L3
        {'params': net.input_hidden_layers[str(len(net.input_hidden_layers)-1)].parameters(), 'lr': 0.2},  # L1
        {'params': net.hidden_output_layers[str(len(net.hidden_output_layers)-1)].parameters(), 'lr': 0.005},  # L2
    ]

    for num in range(net.n_hidden_layers - 1):
        params.append({'params': net.hidden_hidden_layers[str(len(net.hidden_hidden_layers)-num-1)].parameters(), 'lr': 0.005},)  # L2

    optimizer = torch.optim.RMSprop(params, momentum=0.9, weight_decay=0.00001, centered=True)
    return optimizer