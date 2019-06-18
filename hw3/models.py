import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Block, Linear, ReLU, Sigmoid, Dropout, Sequential


class MLP(Block):
    """
    A simple multilayer perceptron model based on our custom Blocks.
    Architecture is (with ReLU activation):

        FC(in, h1) -> ReLU -> FC(h1,h2) -> ReLU -> ... -> FC(hn, num_classes)

    Where FC is a fully-connected layer and h1,...,hn are the hidden layer
    dimensions.
    If dropout is used, a dropout layer is added after every activation
    function.
    """
    def __init__(self, in_features, num_classes, hidden_features=(),
                 activation='relu', dropout=0, **kw):
        super().__init__()
        """
        Create an MLP model Block.
        :param in_features: Number of features of the input of the first layer.
        :param num_classes: Number of features of the output of the last layer.
        :param hidden_features: A sequence of hidden layer dimensions.
        :param activation: Either 'relu' or 'sigmoid', specifying which 
        activation function to use between linear layers.
        :param: Dropout probability. Zero means no dropout.
        """
        blocks = []

        # TODO: Build the MLP architecture as described.
        # ====== YOUR CODE: ======
        if activation=='relu':
            act = ReLU
        else:
            act = Sigmoid
        inf = in_features
        for hidden in hidden_features:
            blocks.append(Linear(inf,hidden))
            blocks.append(act())
            inf = hidden
        if not hidden_features: #in case list is empty
            hidden_features += [in_features]
        blocks.append(Linear(hidden_features[-1] , num_classes)) #last layer to num classes
        # ========================
        self.sequence = Sequential(*blocks)

    def forward(self, x, **kw):
        return self.sequence(x, **kw)

    def backward(self, dout):
        return self.sequence.backward(dout)

    def params(self):
        return self.sequence.params()

    def train(self, training_mode=True):
        self.sequence.train(training_mode)

    def __repr__(self):
        return f'MLP, {self.sequence}'


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(Conv -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """

    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param filters: A list of of length N containing the number of
            filters in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        self.filters = filters
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        # Pooling to reduce dimensions.
        # ====== YOUR CODE: ======
        pool = self.pool_every - 1
        for i , filter in enumerate(self.filters):
            # note that there are very weird options for conv layers, take a look in docs to see what are the options
            # most weird options (such as dialation) are off by default, but worth taking a look
            # chosen 1 padding as this preserves shape with 3*3 filters (here called kernel)
            layers.append(torch.nn.Conv2d(in_channels, filter, kernel_size=3, padding=1))
            layers.append(torch.nn.ReLU())#ReLu
            if (i == pool):
                layers.append(torch.nn.MaxPool2d(kernel_size=2))
                pool += self.pool_every
            in_channels = filter  # this is the number of channels towards next layer
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        # ====== YOUR CODE: ======
        if self.pool_every <= 0 or (self.pool_every > len(self.filters)): #avoid deviding by 0 if there are no pools
            pool_num = 0
        else:
            pool_num = len(self.filters) / self.pool_every
        in_features = (int)(in_h / (2 ** pool_num)) #every pool input size is cut in half
        in_features = (in_features**2) * self.filters[-1] #this matrix will be flattened
        for i, dim in enumerate(self.hidden_dims):
            layers.append(torch.nn.Linear(in_features, dim))
            layers.append(torch.nn.ReLU())
            in_features = dim
        layers.append(torch.nn.Linear(in_features, self.out_classes))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        # Extract features from the input, run the classifier on them and
        # return class scores.
        # ====== YOUR CODE: ======
        f = self.feature_extractor(x)
        f = f.reshape(x.shape[0],-1) #a bit naibourhood as conv layers work on tensors per sample, and fc on vectors
        #print(f.shape)
        out = self.classifier(f)
        # ========================
        return out


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        super().__init__(in_size, out_classes, filters, pool_every, hidden_dims)

    # TODO: Change whatever you want about the ConvClassifier to try to
    # improve it's results on CIFAR-10.
    # For example, add batchnorm, dropout, skip connections, change conv
    # filter sizes etc.
    # ====== YOUR CODE: ======


    #overriding the methods that generate a conv model


    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        # Pooling to reduce dimensions.
        # ====== YOUR CODE: ======
        self.p = 0.1
        layers.append(nn.BatchNorm2d(in_channels))
        pool = self.pool_every - 1
        for i, filter in enumerate(self.filters):
            # chosen 1 padding as this preserves shape with 3*3 filters (here called kernel)
            layers.append(torch.nn.Conv2d(in_channels, filter, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(filter))
            layers.append(torch.nn.ReLU())
            #layers.append(torch.nn.LeakyReLU())
            layers.append(torch.nn.Dropout(self.p))
            if (i == pool):
                layers.append(torch.nn.MaxPool2d(kernel_size=2))
                pool += self.pool_every
            in_channels = filter  # this is the number of channels towards next layer
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        # ====== YOUR CODE: ======
        if self.pool_every <= 0 or (self.pool_every > len(self.filters)):  # avoid deviding by 0 if there are no pools
            pool_num = 0
        else:
            pool_num = len(self.filters) / self.pool_every
        in_features = (int)(in_h / (2 ** pool_num))  # every pool input size is cut in half
        in_features = (in_features ** 2) * self.filters[-1]  # this matrix will be flattened
        for i, dim in enumerate(self.hidden_dims):
            layers.append(torch.nn.Linear(in_features, dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(self.p))
            in_features = dim
        layers.append(torch.nn.Linear(in_features, self.out_classes))
        # ========================
        seq = nn.Sequential(*layers)
        return seq





    # ========================

