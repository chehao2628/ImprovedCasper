# import libraries
import torch.nn as nn
import torch.utils.data
from Dataloader import *
from CasperNetwork import Net
from AddHiddenNeuron import addNeuron
from utils import *


input_size = 10
num_classes = 7
batch_size = 150

# load all data
data = load_data()

# Create torch Dict to store and install neurons to network. See AddHiddenNeuron Class to understand the add neuron
# operation or Casper by Pytorch implementation
input_hidden_layers = nn.ModuleDict()
hidden_hidden_layers = nn.ModuleDict()
hidden_output_layers = nn.ModuleDict()

k = 10
accuracy_list = list()
sample_num = int(674 * ((k - 1) / k))
net_set = list()
for i in range(k):
    # Initilize a epoch number. It will be auto increased if training needs.
    num_epochs = 1200
    # define train dataset and a val set
    train_val_sets = split_Kdata(data, k, i)
    train_set = train_val_sets[0]
    val_set = train_val_sets[1]

    # Normalize data using z-score method
    for data_set in train_val_sets:
        normalization(data_set)
    data_tensor = torch.Tensor(train_set.values)

    # Load model
    net = Net(input_size, num_classes, input_hidden_layers, hidden_hidden_layers, hidden_output_layers)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()  # nn.CrossEntropyLoss() computes softmax internally
    addNeuron(net)

    # The time period is 15+P*N (refer to: A Cascade network algorithm employing Progressive RPROP) where N is
    # the number of currently installed neurons, and P is a parameter set prior to training.)
    P = 2
    # checkpoint_epoch = 15 + P * N
    checkpoint_epoch = 15 + P * net.n_hidden_layers

    # weighted Sampler
    weights = make_weights_for_balanced_classes(train_set, num_classes)
    sampler = WeightedRandomSampler(weights, num_samples=sample_num, replacement=True)
    train_dataset = DataFrameDataset(df=data_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    optimizer = torch.optim.RMSprop(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.00001, centered=True)

    # train the model by batch
    previous_loss = float('inf')
    count = 0
    for epoch in range(num_epochs * 2):

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
                if net.n_hidden_layers > 10:
                    P = 15
                # Modify this if want different maximum number of hidden neurons
                if net.n_hidden_layers >= 12:
                    break
                # Here to add a new Hidden Neron for Casper
                optimizer = addNeuron(net)
                N = net.n_hidden_layers
                if checkpoint_epoch + (15 + P * N) > num_epochs:
                    num_epochs += 15 + P * net.n_hidden_layers
            previous_loss = total_loss
            checkpoint_epoch += 15 + P * N
            print('The number of Hidden Neurons have been added is: ', net.n_hidden_layers)
            print('Epoch [%d/%d], Loss: %.4f, Accuracy: %.2f %%'
                  % (epoch + 1, num_epochs,
                     total_loss, 100 * correct / total))
            if checkpoint_epoch > num_epochs:
                break
    net_set.append(net)

    # print("Epoch comes to..........", epoch, 'and stopped here.')

    """
    Evaluating the Results
    """
    # get testing data
    test_input = val_set.iloc[:, 1:]
    test_target = val_set.iloc[:, 0]
    inputs = torch.Tensor(test_input.values).float()
    targets = torch.Tensor(test_target.values - 1).long()

    outputs = net(inputs)
    _, predicted = torch.max(outputs, 1)

    total = predicted.size(0)
    correct = predicted.data.numpy() == targets.data.numpy()
    accuracy = (sum(correct) / total)
    accuracy_list.append(accuracy)
    print("Accuracy is in fold ", i, ' is: ', 100 * accuracy, "%")

# Estimate the accuracy on test set
print("Std in 10-Fold is", np.std(accuracy_list))
print("Median in 10-Fold is", np.median(accuracy_list))
print("Average accuracy in 10-Fold is", 100 * np.average(accuracy_list), "%")
print("Highest accuracy in 10-Fold is", 100 * np.max(accuracy_list), "%")
