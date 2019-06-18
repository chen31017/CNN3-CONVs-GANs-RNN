import abc
import torch
import math

class Block(abc.ABC):
    """
    A block is some computation element in a network architecture which
    supports automatic differentiation using forward and backward functions.
    """
    def __init__(self):
        # Store intermediate values needed to compute gradients in this hash
        self.grad_cache = {}
        self.training_mode = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        Computes the forward pass of the block.
        :param args: The computation arguments (implementation specific).
        :return: The result of the computation.
        """
        pass

    @abc.abstractmethod
    def backward(self, dout):
        """
        Computes the backward pass of the block, i.e. the gradient
        calculation of the final network output with respect to each of the
        parameters of the forward function.
        :param dout: The gradient of the network with respect to the
        output of this block.
        :return: A tuple with the same number of elements as the parameters of
        the forward function. Each element will be the gradient of the
        network output with respect to that parameter.
        """
        pass

    @abc.abstractmethod
    def params(self):
        """
        :return: Block's trainable parameters and their gradients as a list
        of tuples, each tuple containing a tensor and it's corresponding
        gradient tensor.
        """
        pass

    def train(self, training_mode=True):
        """
        Changes the mode of this block between training and evaluation (test)
        mode. Some blocks have different behaviour depending on mode.
        :param training_mode: True: set the model in training mode. False: set
        evaluation mode.
        """
        self.training_mode = training_mode


class Linear(Block):
    """
    Fully-connected linear layer.
    """

    def __init__(self, in_features, out_features, wstd=0.1):
        """
        :param in_features: Number of input features (Din)
        :param out_features: Number of output features (Dout)
        :wstd: standard deviation of the initial weights matrix
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # TODO: Create the weight matrix (w) and bias vector (b).

        # ====== YOUR CODE: ======
        self.w = torch.randn(out_features,in_features)# / wstd
        self.b = torch.randn(out_features)# / wstd
        # ========================

        self.dw = torch.zeros_like(self.w)
        self.db = torch.zeros_like(self.b)

    def params(self):
        return [
            (self.w, self.dw), (self.b, self.db)
        ]

    def forward(self, x, **kw):
        """
        Computes an affine transform, y = x W^T + b.
        :param x: Input tensor of shape (N,Din) where N is the batch
        dimension, and Din is the number of input features, or of shape
        (N,d1,d2,...,dN) where Din = d1*d2*...*dN.
        :return: Affine transform of each sample in x.
        """

        x = x.reshape((x.shape[0], -1))

        # TODO: Compute the affine transform

        # ====== YOUR CODE: ======
        out = torch.matmul(x,self.w.transpose(0,1))
        out += self.b
        # ========================

        self.grad_cache['x'] = x
        return out

    def backward(self, dout):
        """
        :param dout: Gradient with respect to block output, shape (N, Dout).
        :return: Gradient with respect to block input, shape (N, Din)
        """
        x = self.grad_cache['x']

        # TODO: Compute
        #   - dx, the gradient of the loss with respect to x
        #   - dw, the gradient of the loss with respect to w
        #   - db, the gradient of the loss with respect to b
        # You should accumulate gradients in dw and db.
        # ====== YOUR CODE: ======
        # gradient for W:
        # we can observe that the gradient with respect to w^T (knowing backprop for matrix mul.) is:
        # dL / dw^t = x^T * dout. realizing that the gradient for the transposed is the transposed matrix we get:
        # dL / dw^T = (dout)^T * x
        self.dw = torch.matmul(dout.transpose(0,1),x)
        # Gradient for bias:
        # this is an addition operation, so intuitively the derivative is 1. working with matrices complicates a bit.
        # for any coordinate in (x*W^T + b) the derivative with regard to (b)i is
        # 1 if that coordinate is in the i'th row, 0 otherwise.
        # so we get a *very* large Jacobian full of zeros and 1's. As we need to multiply it with the upstream grad
        # we'll skip calculating it and just remember that this is all going to be multiplied with the downstream
        # gradient which will get us back to a normal size
        self.db = torch.matmul(torch.ones(dout.shape[0]),dout)
        #similarly for w, the gradient with regards to x is: (dout * (w^T)^T) = dout* w
        dx = torch.matmul(dout,self.w)
        #raise NotImplementedError()
        # ========================

        return dx

    def __repr__(self):
        return f'Linear({self.in_features}, {self.out_features})'


class ReLU(Block):
    """
    Rectified linear unit.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, **kw):
        """
        Computes max(0, x).
        :param x: Input tensor of shape (N,*) where N is the batch
        dimension, and * is any number of other dimensions.
        :return: ReLU of each sample in x.
        """

        # TODO: Implement the ReLU operation.
        # ====== YOUR CODE: ======
        out = torch.max(input=x, other=torch.zeros_like(x), out=None)
        #out = torch.nn.functional.relu(x)
        # ========================

        self.grad_cache['x'] = x
        return out

    def backward(self, dout):
        """
        :param dout: Gradient with respect to block output, shape (N, *).
        :return: Gradient with respect to block input, shape (N, *)
        """
        x = self.grad_cache['x']

        # TODO: Implement gradient w.r.t. the input x
        # ====== YOUR CODE: ======
        # Again, the local gradient might be huge, but it's a diagonal matrix. and thinking even further, it's easy to
        # see that any negative value in x has no affect on downstream value, while for any positive value this is just
        # the identity function (this is the most intuitive way to consider this, it's also possible to simulate the
        # multipication process).
        mask = torch.zeros(x.shape[0], x.shape[1])  # auxilary: tensor.where demands a tensor, we'll use this one
        # turn matrix binary - every cell smaller than 0 turns to 0 (since we didn't have negatives we are left with 0's)
        dx = torch.where(dout < 0, mask, dout)
        # ========================

        return dx

    def params(self):
        return []

    def __repr__(self):
        return 'ReLU'


class Sigmoid(Block):
    """
    Sigmoid activation function.
    """
    def __init__(self):
        super().__init__()


    def forward(self, x, **kw):
        """
        Computes s(x) = 1/(1+exp(-x))
        :param x: Input tensor of shape (N,*) where N is the batch
        dimension, and * is any number of other dimensions.
        :return: Sigmoid of each sample in x.
        """

        # TODO: Implement the Sigmoid function. Save whatever you need into
        # grad_cache.
        # ====== YOUR CODE: ======
        tmp = torch.sigmoid(x) #debug
        out = 1 / (1 + torch.pow(math.e, -x))
        #my implementation passed testes but when working in sequential mode, there were very small diffs (about
        # 1e^15 that were magnified upstream causing major calculation errors upstream. sent a mail to Tomer, meanwhile
        # Using the frameworks builtin function
        #print (tmp - out) #this gave almost 0 dif

        out = tmp

        self.grad_cache['s'] = out
        # ========================

        return out

    def backward(self, dout):
        """
        :param dout: Gradient with respect to block output, shape (N, *).
        :return: Gradient with respect to block input, shape (N, *)
        """

        # TODO: Implement gradient w.r.t. the input x
        # ====== YOUR CODE: ======
        s = self.grad_cache['s']
        tmp = torch.mul((1-s),s) #elementwise derivative
        self.grad_cache['s'] = s #repare damage to the cached value
        dx = torch.mul(tmp,dout) #each coordinate in partial gradient effects only the corasponding element in dout
        # ========================

        return dx

    def params(self):
        return []

    def __repr__(self):
        return 'Sigmoid'


class CrossEntropyLoss(Block):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        """
        Computes cross-entropy loss directly from class scores.
        Given class scores x, and a 1-hot encoding of the correct class yh,
        the cross entropy loss is defined as: -yh^T * log(softmax(x)).

        This implementation works directly with class scores (x) and labels
        (y), not softmax outputs or 1-hot encodings.

        :param x: Tensor of shape (N,D) where N is the batch
        dimension, and D is the number of features. Should contain class
        scores, NOT PROBABILITIES.
        :param y: Tensor of shape (N,) containing the ground truth label of
        each sample.
        :return: Cross entropy loss, as if we computed the softmax of the
        scores, encoded y as 1-hot and calculated cross-entropy by
        definition above. A scalar.
        """

        N = x.shape[0]
        xmax, _ = torch.max(x, dim=1, keepdim=True)
        x = x - xmax  # for numerical stability

        # TODO: Compute the cross entropy loss using the last formula from the
        # notebook (i.e. directly using the class scores).
        # Tip: to get a different column from each row of a matrix tensor m,
        # you can index it with m[range(num_rows), list_of_cols].
        # ====== YOUR CODE: ======
        # get the exponent of all scores.
        # exp = torch.pow(math.e, x, out=None)
        # sum each row.
        # take log of each row
        logs = torch.logsumexp(x, 1, keepdim=False, out=None)
        correct = x[range(N), y]
        out = logs - correct
        loss = torch.sum(out) / N
        # ========================
        self.grad_cache['x'] = x
        self.grad_cache['y'] = y
        return loss

    def backward(self, dout=1.0):
        """
        :param dout: Gradient with respect to block output, a scalar which
        defaults to 1 since the output of forward is scalar.
        :return: Gradient with respect to block input (only x), shape (N,D)
        """
        x = self.grad_cache['x']
        y = self.grad_cache['y']
        N = x.shape[0]

        # TODO: Calculate the gradient w.r.t. the input x
        # ====== YOUR CODE: ======
        # The derivative changes between the correct class score and the incorrect:
        #   for incorrect class score: d(loss) / d(x)ij = 1/N(e^(x)ij / sum(e^(x))) for all scores in sample
        #   for the correct class score we get the same expression but with added -1/N
        exp = torch.pow(math.e, x, out=None)
        sumexp = 1 / torch.sum(exp,dim=1) #inverce of sum of exponents for each sample
        dx = 1/N * torch.mul(exp,sumexp.unsqueeze(1))
        #still need to deduct 1/N from each "correct" score's derivative
        dx[range(N), y] = dx[range(N), y] - 1/N
        # ========================
        return dx

    def params(self):
        return []


class Dropout(Block):
    def __init__(self, p=0.5):
        """
        Initializes a Dropout block.
        :param p: Probability to drop an activation.
        """
        super().__init__()
        assert 0. <= p <= 1.
        self.p = p

    def forward(self, x, **kw):
        # TODO: Implement the dropout forward pass. Notice that contrary to
        # previous blocks, this block behaves differently a according to the
        # current mode (train/test).
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================

        return out

    def backward(self, dout):
        # TODO: Implement the dropout backward pass.
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================

        return dx

    def params(self):
        return []

    def __repr__(self):
        return f'Dropout(p={self.p})'


class Sequential(Block):
    """
    A Block that passes input through a sequence of other blocks.
    """
    def __init__(self, *blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x, **kw):
        out = None

        # TODO: Implement the forward pass by passing each block's output
        # as the input of the next.
        # ====== YOUR CODE: ======
        out = x
        for i, block in enumerate(self.blocks):
            if isinstance(block, CrossEntropyLoss):
                out = block.forward(out,**kw)
                continue
            out = block.forward(out)
        # ========================
        return out

    def backward(self, dout):
        din = None

        # TODO: Implement the backward pass.
        # Each block's input gradient should be the previous block's output
        # gradient. Behold the backpropagation algorithm in action!
        # ====== YOUR CODE: ======
        din = dout
        for block in reversed(self.blocks): #reversed as we want to work backwards through the net
            din = block.backward(din)
            #print("gradient sum for this layer" ,  din.mean())
        # ========================

        return din

    def params(self):
        params = []

        # TODO: Return the parameter tuples from all blocks.
        # ====== YOUR CODE: ======
        for block in self.blocks:
            params += (block.params())
        # ========================

        return params

    def train(self, training_mode=True):
        for block in self.blocks:
            block.train(training_mode)

    def __repr__(self):
        res = 'Sequential\n'
        for i, block in enumerate(self.blocks):
            res += f'\t[{i}] {block}\n'
        return res

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, item):
        return self.blocks[item]

