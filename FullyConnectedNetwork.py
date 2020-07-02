"""
This script provides an example of building a neural
network for classifying glass identification dataset on
http://archive.ics.uci.edu/ml/datasets/Glass+Identification
It loads data using a data loader, and trains a neural
network with batch training.
"""

# import libraries
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from Dataloader import *
from utils import *

# Hyper Parameters
input_size = 10
num_classes = 7
num_epochs = 200
batch_size = 100
learning_rate = 0.01

# load all data
data = load_data()


# Fully Connected Neural Network
class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        # Modify this tod test different fully connected network sizes
        self.fc1 = nn.Linear(input_size, 12)
        # self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(12, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.leaky_relu(out)
        # out = self.fc2(out)
        # out = F.leaky_relu(out)
        out = self.fc3(out)
        return out


# k-fold cross validate
k = 10
accuracy_list = list()
precision_list = list()
recall_list = list()
sample_num = int(674 * ((k - 1) / k))

for i in range(k):
    # define train dataset and a data loader
    train_val_sets = split_Kdata(data, k, i)
    train_set = train_val_sets[0]
    val_set = train_val_sets[1]
    # Normalize data using z-score method
    for data_set in train_val_sets:
        normalization(data_set)
    data_tensor = torch.Tensor(train_set.values)

    net = Net(input_size, num_classes)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()  # nn.CrossEntropyLoss() computes softmax internally
    optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate)

    weights = make_weights_for_balanced_classes(train_set, num_classes)
    sampler = WeightedRandomSampler(weights, num_samples=sample_num, replacement=True)
    train_dataSet = DataFrameDataset(df=data_tensor, noisy=False)
    train_loader = torch.utils.data.DataLoader(train_dataSet, batch_size=batch_size, sampler=sampler)

    all_losses = []
    # train the model by batch
    for epoch in range(num_epochs):
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

            if (epoch % 50 == 0):
                prob_predicted = torch_max(outputs)
                _, predicted = torch.max(outputs, 1)
                for t in range(len(predicted)):
                    if prob_predicted[t] == 1:
                        prob_predicted[t] = predicted[t]
                prob_predicted = torch.Tensor(prob_predicted)
                # print("ssssss", prob_predicted)

                # calculate and print accuracy
                total = total + predicted.size(0)
                correct = correct + sum(predicted.data.numpy() == Y.data.numpy())
                total_loss = total_loss + loss

        if (epoch % 50 == 0):
            c = 0
            for p in prob_predicted:
                if p == -1:
                    c += 1
            print('Epoch [%d/%d], Loss: %.4f, Accuracy: %.2f %%'
                  % (epoch + 1, num_epochs,
                     total_loss, 100 * correct / (total-c)))




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

#     """
#     Evaluating the Results
#     """
#     # get testing data
#     test_input = val_set.iloc[:, 1:]
#     test_target = val_set.iloc[:, 0]
#     inputs = torch.Tensor(test_input.values).float()
#     targets = torch.Tensor(test_target.values - 1).long()
#
#     outputs = net(inputs)
#     prob_predicted = torch_max(outputs)
#     reverse_prob_predicted = revers_torch_max(outputs)
#     _, predicted = torch.max(outputs, 1)
#     for t in range(len(predicted)):
#         if prob_predicted[t] == 1:
#             prob_predicted[t] = predicted[t]
#         else:
#             reverse_prob_predicted[t] = predicted[t]
#     prob_predicted = torch.Tensor(prob_predicted)
#     reverse_prob_predicted = torch.Tensor(reverse_prob_predicted)
#     reverse_total = reverse_prob_predicted.size(0)
#     # count the number of tn+fn
#     pred_unclassify = 0  # pred_unclassify = TN + FN
#     for p in prob_predicted:
#         if p == -1:
#             pred_unclassify += 1
#     correct = prob_predicted.data.numpy() == targets.data.numpy()
#     incorrect = reverse_prob_predicted.data.numpy() == targets.data.numpy()
#     Total = prob_predicted.size(0)
#     TP = sum(correct)
#     FP = Total - pred_unclassify -TP
#     FN = sum(incorrect)
#     TN = pred_unclassify - FN
#     print(TP, FP, FN, TN)
#     accuracy = (TP + TN) / Total
#     precision = TP / (TP + FP)
#     recall = TP / (TP + FN)
#     specificity = TN /(FP + TN)
#
#     accuracy_list.append(accuracy)
#     precision_list.append(precision)
#     recall_list.append(recall)
#     print("Accuracy is in fold ", i, ' is: ', 100 * accuracy, "%")
#     print("Precision is in fold ", i, ' is: ', 100 * precision, "%")
#     print("Recall is in fold ", i, ' is: ', 100 * recall, "%")
#     print("Specificity is in fold ", i, ' is: ', 100 * specificity, "%")
#
#
#
# print("Std in 10-Fold is", np.std(accuracy_list))
# print("Mediany in 10-Fold is", np.median(accuracy_list))
# print("Average accuracy in 10-Fold is", 100 * np.average(accuracy_list), "%")
# print("Precision accuracy in 10-Fold is", 100 * np.average(precision_list), "%")
# print("Recall accuracy in 10-Fold is", 100 * np.average(recall_list), "%")


