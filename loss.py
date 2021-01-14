from torch.nn import Module
import torch

import torch.nn.functional as F


class BiTemperedLogisticLoss(Module):
    """
    Implements bi-tempered logistic loss as per https://arxiv.org/pdf/1906.03361.pdf
    Helps to train and generalise in the presence of noisy data, which is true in cassava competition
    """

    def __init__(self, t1 = 0.5, t2 = 4., *args, **kwargs):
        """
        Initiliase class with temperature params
        :param args:
        :param t1: Temperature 1
        :param t2: Temperature 2
        """
        super().__init__(*args, **kwargs)
        self.t1 = t1
        self.t2 = t2

    def log_t(self, x, t):
        """
        Returns the tempered log function described in paper
        :param x: The variable to be operated on with the log function
        :param t: The temperature parameter
        :return:
        """

        # avoid divide by zero
        if t != 1.:
            return (1. / (1. - t)) * (x ** (1. - t) - 1.)
        else:
            return torch.log(x)

    def exp_t(self, x, t):
        """
        Returns the tempered exponential function described in paper
        :param x: The variable to be operated on with the exp function
        :param t: The temperature parameter
        :return:
        """

        # avoid divide by zero
        if t != 1.:
            return F.relu((1. - t) * x  + 1.)** (1. / (1. - t))
        else:
            return torch.exp(x)

    def lambda_t_a(self, a, t, epsilon=0.001):
        """
        Returns the normalisation \lambda_t(a)
        :param a: Tensor of activations
        :param t: Temperature parameter (requires t>1 for this algorithm)
        :param epsilon: Threshold to consider the algorithm to have converged
        :return:
        """
        if t > 1.:

            # follow algorithm from Appendix A

            mu = torch.max(a,dim=-1, keepdim=True).values

            norm_a_initial = a - mu

            norm_a = norm_a_initial

            last_norm_a = torch.zeros(norm_a.shape).cuda()

            while torch.linalg.norm(norm_a - last_norm_a) > epsilon:
                z_a = torch.sum(self.exp_t(norm_a, t), dim=-1, keepdim=True)
                last_norm_a = norm_a
                norm_a = (z_a ** (1. - t)) * norm_a_initial


            z_a = torch.sum(self.exp_t(norm_a, t), dim=-1, keepdim=True)

            return -1. * self.log_t((1. / z_a), t) + mu

        elif t < 1.:
            # Follow binary search procedure, loosely translated from https://github.com/google/bi-tempered-loss/blob/master/tensorflow/loss.py

            mu = torch.max(a,dim=-1, keepdim=True).values
            norm_a_initial = a - mu

            a_shape = a.shape

            # not sure what effective dim is meant to mean but thats the starting name for the upper bound in the Google code
            effective_dim = torch.sum(((-1. / (1. - t)) * torch.ones(a_shape) < norm_a_initial).int(), dim=-1, keepdim=True).float()

            shape_partition = torch.cat((a_shape[:-1], torch.tensor([1])), 0)

            lower = torch.zeros(shape_partition)
            upper = -1. * self.log_t((1. / effective_dim), t) * torch.ones(shape_partition)

            while torch.linalg.norm(upper-lower) > epsilon:

                logt_partition = (upper + lower) / 2.

                prob_sum = torch.sum(self.exp_t((norm_a_initial - logt_partition), t), dim=-1, keepdim=True)

                increment = (prob_sum < 1.).float()

                lower = torch.reshape((lower * increment + (1. - increment) * logt_partition), shape_partition)

                upper = torch.reshape((upper * (1. - increment)  + increment * logt_partition), shape_partition)

            logt_partition = (upper + lower) / 2.

            return logt_partition + mu

        else:
            # If t==1, normalisation is just as normal for log loss
            return torch.log(torch.sum(torch.exp(a), dim=-1, keepdim=True))

    def forward(self, inputs, targets):
        """
        The actual loss function pass
        :param inputs: The network outputs passed to lass function
        :param targets: The target labels
        :return:
        """

        norm_constants = self.lambda_t_a(inputs, self.t2)

        probabilities = self.exp_t(inputs-norm_constants, self.t2)

        loss1 = targets * (self.log_t(targets + 1e-8, self.t1) - self.log_t(probabilities, self.t1))

        loss2 = (1. / (2. - self.t1)) * (torch.pow(targets, 2. - self.t1) - torch.pow(probabilities, 2. - self.t1))

        loss = loss1 - loss2

        out = torch.sum(loss, dim=-1)

        return torch.mean(out)










