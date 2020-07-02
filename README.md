# ImprovedCasper
This is a individual project. A structure 'Improved Casper' based on the Casper (Treadgold and Gedeon, 1997) Algorithm. Casper is an improved version of Cascor (Cascade Correlation) which is a Neural Network structure (Fahlman and Lebiere, 1990) is produced. The techniques used in original Casper algorithm are old claimed 20 years ago. In this project, I applied state-of-art techniques to improve the Casper Algorithm. The Dataset SFEW(Static Facial Expressions in the Wild) was used to evaluate Improved Casper. This project implemented with Pytorch.

Input of Network: 10 classes
output of Network: 7 classes

Run FullyConnectedNetwork.py to measure Fully connected neural network with K fold validation on Dataset.
Run KfoldMain.py to measure improved Casper with K fold validation on SFEW Dataset.

Note: Run KfoldMain.py is high time cost. Run SimpleMain.py can test without cross validation.

Techniques are implemented as following:

Add Noise: in Class DataFrameDataset() of python file Dataloader.py
Z-score transformation: in Function normalization() of python file Dataloader.py
Weighted Sampler: in Function make_weights_for_balanced_classes of python file Dataloader.py
Techniques in Casper Model Training include Leakey ReLU, Cross Entropy Error function, RMSprop and Epoch limitation
release are integrated in KfoldMain.py, KfoldMain.py and FullyConnectedNetwork.py.
