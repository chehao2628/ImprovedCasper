import torch.nn as nn
import torch.nn.functional as F

# Neural Network
class Net(nn.Module):
    def __init__(self, input_size, num_classes, input_hidden_layers, hidden_hidden_layers, hidden_output_layers):
        super().__init__()
        self.n_hidden_layers = 0
        self.fc1 = nn.Linear(input_size, num_classes)
        # self.bn_input = nn.BatchNorm1d(10, momentum=0.9)

        # Init number of n*(n+3)/2 layers for use
        self.input_hidden_layers = input_hidden_layers
        self.hidden_hidden_layers = hidden_hidden_layers
        self.hidden_output_layers = hidden_output_layers

    def forward(self, x):
        outL3_1 = self.fc1(x)  # part of L3 weights, correlation of input and output
        if self.n_hidden_layers == 0:
            return outL3_1

        H = list()  # store connections of input classes and all hidden units (L1 weights and part of L3 weights)

        # store the first connection (between input and first hidden unit)
        H.append(F.leaky_relu(self.input_hidden_layers['0'](x)))

        if self.n_hidden_layers == 1:
            for h in H:
                # Get connections related to L2 Weights
                outL2 = self.hidden_output_layers['0'](h)
                return outL3_1 + outL2

        # if n_hidden_layers>1, do the following iteration
        count1 = 0  # record the index of hidden_hidden_layers
        for i in range(1, self.n_hidden_layers):
            # build the current hidden unit, init with self.input_hidden
            current_hidden_unit = F.leaky_relu(self.input_hidden_layers[str(i)](x))
            c_list = list()
            c_list.append(current_hidden_unit)
            for h in H:
                # if len(H)-count1 > 3:
                #     previous_connection.detach()
                current_hidden_unit += F.leaky_relu(self.hidden_hidden_layers[str(count1)](h))
                count1 += 1
            H.append(current_hidden_unit)

        # Connect hidden unit to output
        total_out = outL3_1
        count2 = 0  # record the index of hidden_output_layers
        for h in H:
            total_out = total_out + self.hidden_output_layers[str(count2)](h)
            count2 += 1
        return total_out
