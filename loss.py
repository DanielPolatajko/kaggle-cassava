from torch.nn import Module
import torch

import torch.nn.functional as F

import numpy as np


class BiTemperedLogisticLoss(Module):
    """
    Implements bi-tempered logistic loss as per https://arxiv.org/pdf/1906.03361.pdf
    Helps to train and generalise in the presence of noisy data, which is true in cassava competition
    """

    def __init__(self, t1 = 0.2, t2 = 1.2, label_smoothing=0.05, *args, **kwargs):
        """
        Initiliase class with temperature params
        :param args:
        :param t1: Temperature 1
        :param t2: Temperature 2
        """
        super().__init__(*args, **kwargs)
        self.t1 = t1
        self.t2 = t2
        self.label_smoothing = label_smoothing

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
            return F.relu((1. - t) * x  + 1.) ** (1. / (1. - t))
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

        if self.label_smoothing > 0.0:
            num_classes = targets.shape[-1]
            targets = (1 - num_classes / (num_classes - 1) * self.label_smoothing) * targets + self.label_smoothing / (
                        num_classes - 1)

        norm_constants = self.lambda_t_a(inputs, self.t2)

        probabilities = self.exp_t(inputs-norm_constants, self.t2)

        loss1 = targets * (self.log_t(targets + 1e-8, self.t1) - self.log_t(probabilities, self.t1))

        loss2 = (1. / (2. - self.t1)) * (torch.pow(targets, 2. - self.t1) - torch.pow(probabilities, 2. - self.t1))

        loss = loss1 - loss2

        out = torch.sum(loss, dim=-1)

        return torch.mean(out)

"""
Snapmix augmentation and loss, loosely from https://www.kaggle.com/sachinprabhu/pytorch-resnet50-snapmix-train-pipeline
"""

class SnapMix:
    """
    Class implementing the snapmix augmentation https://arxiv.org/abs/2012.04846
    """
    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def get_spm(self, input, target, model, img_size):
        imgsize = (img_size, img_size)
        bs = input.size(0)
        with torch.no_grad():
            output, fms = model(input)
            clsw = model.fc.linear
            weight = clsw.weight.data
            bias = clsw.bias.data
            weight = weight.view(weight.size(0), weight.size(1), 1, 1)
            fms = F.relu(fms)
            poolfea = F.adaptive_avg_pool2d(fms, (1, 1)).squeeze()
            clslogit = F.softmax(clsw.forward(poolfea), dim=-1)

            # deal with the case where last batch has size 1
            if len(clslogit.shape) == 1:
                clslogit = torch.unsqueeze(clslogit, 0)

            logitlist = []
            for i in range(bs):
                logitlist.append(clslogit[i, (target[i]).nonzero(as_tuple=False).item()])

            clslogit = torch.stack(logitlist)

            out = F.conv2d(fms, weight, bias=bias)

            outmaps = []
            for i in range(bs):
                evimap = out[i, (target[i]).nonzero(as_tuple=False).item()]
                outmaps.append(evimap)

            outmaps = torch.stack(outmaps)

            if imgsize is not None:
                outmaps = outmaps.view(outmaps.size(0), 1, outmaps.size(1), outmaps.size(2))
                outmaps = F.interpolate(outmaps, imgsize, mode='bilinear', align_corners=False)

            outmaps = outmaps.squeeze()

            # deal with the case where last batch has size 1
            if len(outmaps.shape) == 2:
                outmaps = torch.unsqueeze(outmaps, 0)

            for i in range(bs):
                outmaps[i] -= outmaps[i].min()
                outmaps[i] /= outmaps[i].sum()

        return outmaps, clslogit

    def __call__(self, input, target, alpha, img_size, model=None):

        r = np.random.rand(1)
        lam_a = torch.ones(input.size(0))
        lam_b = 1 - lam_a
        target_b = target.clone()

        if True:
            wfmaps, _ = self.get_spm(input, target, model, img_size)
            bs = input.size(0)
            lam = np.random.beta(alpha, alpha)
            lam1 = np.random.beta(alpha, alpha)
            rand_index = torch.randperm(bs).cuda()
            wfmaps_b = wfmaps[rand_index, :, :]
            target_b = target[rand_index]

            same_label = torch.equal(target, target_b)
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(input.size(), lam)
            bbx1_1, bby1_1, bbx2_1, bby2_1 = self.rand_bbox(input.size(), lam1)

            area = (bby2 - bby1) * (bbx2 - bbx1)
            area1 = (bby2_1 - bby1_1) * (bbx2_1 - bbx1_1)

            if area1 > 0 and area > 0:
                ncont = input[rand_index, :, bbx1_1:bbx2_1, bby1_1:bby2_1].clone()
                ncont = F.interpolate(ncont, size=(bbx2 - bbx1, bby2 - bby1), mode='bilinear', align_corners=True)
                input[:, :, bbx1:bbx2, bby1:bby2] = ncont
                lam_a = 1 - wfmaps[:, bbx1:bbx2, bby1:bby2].sum(2).sum(1) / (wfmaps.sum(2).sum(1) + 1e-8)
                lam_b = wfmaps_b[:, bbx1_1:bbx2_1, bby1_1:bby2_1].sum(2).sum(1) / (wfmaps_b.sum(2).sum(1) + 1e-8)
                tmp = lam_a.clone()
                lam_a[same_label] += lam_b[same_label]
                lam_b[same_label] += tmp[same_label]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
                lam_a[torch.isnan(lam_a)] = lam
                lam_b[torch.isnan(lam_b)] = 1 - lam

        return input, target, target_b, lam_a.cuda(), lam_b.cuda()


class SnapMixLoss(Module):
    """
    Class implementing the snapmix loss function https://arxiv.org/abs/2012.04846
    """
    def __init__(self):
        super().__init__()

    def forward(self, criterion, outputs, ya, yb, lam_a, lam_b):
        loss_a = criterion(outputs, ya)
        loss_b = criterion(outputs, yb)
        loss = torch.mean(loss_a * lam_a + loss_b * lam_b)
        return loss


# implementations reference - https://github.com/CoinCheung/pytorch-loss/blob/master/pytorch_loss/taylor_softmax.py
# paper - https://www.ijcai.org/Proceedings/2020/0305.pdf

class TaylorSoftmax(Module):

    def __init__(self, dim=1, n=2):
        super(TaylorSoftmax, self).__init__()
        assert n % 2 == 0
        self.dim = dim
        self.n = n

    def forward(self, x):
        fn = torch.ones_like(x)
        denor = 1.
        for i in range(1, self.n + 1):
            denor *= i
            fn = fn + x.pow(i) / denor
        out = fn / fn.sum(dim=self.dim, keepdims=True)
        return out


class LabelSmoothingLoss(Module):

    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        """Taylor Softmax and log are already applied on the logits"""
        # pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class TaylorCrossEntropyLoss(Module):

    def __init__(self, n=2, ignore_index=-1, reduction='mean', smoothing=0.2):
        super(TaylorCrossEntropyLoss, self).__init__()
        assert n % 2 == 0
        self.taylor_softmax = TaylorSoftmax(dim=1, n=n)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lab_smooth = LabelSmoothingLoss(5, smoothing=smoothing)

    def forward(self, logits, labels):
        log_probs = self.taylor_softmax(logits).log()
        # loss = F.nll_loss(log_probs, labels, reduction=self.reduction,
        #        ignore_index=self.ignore_index)
        loss = self.lab_smooth(log_probs, labels)
        return loss










