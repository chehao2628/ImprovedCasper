# import libraries
import torch.nn as nn
import torch.utils.data
from Dataloader import *
from CasperNetwork import Net
from AddHiddenNeuron import addNeuron
from utils import *

input_size = 10
num_classes = 7
num_epochs = 1200
batch_size = 100

# load all data
data = load_data()
# normalization(data)
# define train dataset and a data loader
train_data, test_data = split_data(data)

# Normalize data using z-score method
normalization(train_data)
normalization(test_data)

data_tensor = torch.Tensor(train_data.values)

input_hidden_layers = nn.ModuleDict()
hidden_hidden_layers = nn.ModuleDict()
hidden_output_layers = nn.ModuleDict()
net = Net(input_size, num_classes, input_hidden_layers, hidden_hidden_layers, hidden_output_layers)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()  # nn.CrossEntropyLoss() computes softmax internally
addNeuron(net)

# train the model by batch
previous_loss = float('inf')

# The time period is 15+P*N (refer to: A Cascade network algorithm employing Progressive RPROP) where N is
# the number of currently installed neurons, and P is a parameter set prior to training.)
P = 1
# checkpoint_epoch = 15 + P * N
checkpoint_epoch = 15 + P * net.n_hidden_layers

weights = make_weights_for_balanced_classes(train_data, num_classes)
sampler = WeightedRandomSampler(weights, num_samples=539,replacement=True)
train_dataset = DataFrameDataset(df=data_tensor, noisy=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

# store all losses for visualisation
all_losses = []
optimizer = torch.optim.RMSprop(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.00001, centered=True)

count = 0
for epoch in range(num_epochs*2):
    total = 0
    correct = 0
    total_loss = 0
    for step, (image_feature, labels) in enumerate(train_loader):
        X = image_feature
        Y = labels.long()

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(X)
        loss = criterion(outputs, Y)
        all_losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if (epoch == checkpoint_epoch):
            _, predicted = torch.max(outputs, 1)
            # calculate and print accuracy
            total = total + predicted.size(0)
            correct = correct + sum(predicted.data.numpy() == Y.data.numpy())
            total_loss = total_loss + loss

    if (epoch == checkpoint_epoch):
        # If the loss is no longer decreasing, add a new hidden neuron
        # and set different learning rate to different layers.
        N = net.n_hidden_layers
        if previous_loss < total_loss and net.n_hidden_layers < 100:
            if net.n_hidden_layers > 14:
                P = 15
            # Modify this if want different maximum number of hidden neurons
            if net.n_hidden_layers >= 16:
                break
            # Here to add a new Hidden Neron for Casper
            optimizer = addNeuron(net)
            N = net.n_hidden_layers
            if checkpoint_epoch + (15 + P * N) > num_epochs:
                num_epochs += 15 + P * net.n_hidden_layers

        print('The number of Hidden Neurons have been added is: ', net.n_hidden_layers)
        # print(net.input_hidden_layers[-1])
        # for num in range(net.n_hidden_layers - 1):
        #     print(net.hidden_hidden_layers[-num])
        # print(net.hidden_output_layers[-1])
        previous_loss = total_loss
        print('Epoch [%d/%d], Loss: %.4f, Accuracy: %.2f %%'
              % (epoch + 1, num_epochs,
                 total_loss, 100 * correct / total))
        checkpoint_epoch += 15 + P * N
        print(checkpoint_epoch,"here is checkpoint_epoch")
        if checkpoint_epoch > num_epochs:
            break

print("Epoch comes to..........",epoch, 'and stopped here.')

"""
The following part include plot loss trends and evaluate the on test set in form of Confusion Matrix
The following parts refer to lab2 codes in COMP8420
"""
# Optional: plotting historical loss from ``all_losses`` during network learning
# Please uncomment me from next line to ``plt.show()`` if you want to plot loss
# import matplotlib.pyplot as plt
#
# plt.figure()
# plt.plot(all_losses)
# plt.show()

"""
Evaluating the Results

To see how well the network performs on different categories, we will
create a confusion matrix, indicating for every glass (rows)
which class the network guesses (columns).

"""

train_input = train_data.iloc[:, 1:]
train_target = train_data.iloc[:, 0]

inputs = torch.Tensor(train_input.values).float()
targets = torch.Tensor(train_target.values - 1).long()

outputs = net(inputs)

_, predicted = torch.max(outputs, 1)

print('Confusion matrix for training:')
print(plot_confusion(train_input.shape[0], num_classes, predicted.long().data, targets.data))

"""
Step 3: Test the neural network

Pass testing data to the built neural network and get its performance
"""
# get testing data
test_input = test_data.iloc[:, 1:]
test_target = test_data.iloc[:, 0]
inputs = torch.Tensor(test_input.values).float()
targets = torch.Tensor(test_target.values - 1).long()

outputs = net(inputs)
_, predicted = torch.max(outputs, 1)

total = predicted.size(0)
correct = predicted.data.numpy() == targets.data.numpy()

print('Testing Accuracy: %.2f %%' % (100 * sum(correct) / total))

"""
Evaluating the Results

To see how well the network performs on different categories, we will
create a confusion matrix, indicating for every glass (rows)
which class the network guesses (columns).

"""

print('Confusion matrix for testing:')
print(plot_confusion(test_input.shape[0], num_classes, predicted.long().data, targets.data))
