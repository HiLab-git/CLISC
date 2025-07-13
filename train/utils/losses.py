import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch import einsum






def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                       keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent


def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss


def softmax_kl_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_log_softmax = torch.log(torch.sigmoid(input_logits))
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='mean')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class pDLoss(nn.Module):
    def __init__(self, n_classes, ignore_index):
        super(pDLoss, self).__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, ignore_mask):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target * ignore_mask)
        y_sum = torch.sum(target * target * ignore_mask)
        z_sum = torch.sum(score * score * ignore_mask)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None):
        ignore_mask = torch.ones_like(target)
        ignore_mask[target == self.ignore_index] = 0
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i], ignore_mask)
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def entropy_minmization(p):
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)
    ent = torch.mean(y1)

    return ent


def entropy_map(p):
    ent_map = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                           keepdim=True)
    return ent_map


class SizeLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(SizeLoss, self).__init__()
        self.margin = margin

    def forward(self, output, target):
        output_counts = torch.sum(torch.softmax(output, dim=1), dim=(2, 3))
        target_counts = torch.zeros_like(output_counts)
        for b in range(0, target.shape[0]):
            elements, counts = torch.unique(
                target[b, :, :, :, :], sorted=True, return_counts=True)
            assert torch.numel(target[b, :, :, :, :]) == torch.sum(counts)
            target_counts[b, :] = counts

        lower_bound = target_counts * (1 - self.margin)
        upper_bound = target_counts * (1 + self.margin)
        too_small = output_counts < lower_bound
        too_big = output_counts > upper_bound
        penalty_small = (output_counts - lower_bound) ** 2
        penalty_big = (output_counts - upper_bound) ** 2
        # do not consider background(i.e. channel 0)
        res = too_small.float()[:, 1:] * penalty_small[:, 1:] + \
            too_big.float()[:, 1:] * penalty_big[:, 1:]
        loss = res / (output.shape[2] * output.shape[3] * output.shape[4])
        return loss.mean()


class MumfordShah_Loss(nn.Module):
    def levelsetLoss(self, output, target, penalty='l1'):
        # input size = batch x 1 (channel) x height x width
        outshape = output.shape
        tarshape = target.shape
        self.penalty = penalty
        loss = 0.0
        for ich in range(tarshape[1]):
            target_ = torch.unsqueeze(target[:, ich], 1)
            target_ = target_.expand(
                tarshape[0], outshape[1], tarshape[2], tarshape[3])
            pcentroid = torch.sum(target_ * output, (2, 3)
                                  ) / torch.sum(output, (2, 3))
            pcentroid = pcentroid.view(tarshape[0], outshape[1], 1, 1)
            plevel = target_ - \
                pcentroid.expand(
                    tarshape[0], outshape[1], tarshape[2], tarshape[3])
            pLoss = plevel * plevel * output
            loss += torch.sum(pLoss)
        return loss

    def gradientLoss2d(self, input):
        dH = torch.abs(input[:, :, 1:, :] - input[:, :, :-1, :])
        dW = torch.abs(input[:, :, :, 1:] - input[:, :, :, :-1])
        if self.penalty == "l2":
            dH = dH * dH
            dW = dW * dW

        loss = torch.sum(dH) + torch.sum(dW)
        return loss

    def forward(self, image, prediction):
        loss_level = self.levelsetLoss(image, prediction)
        loss_tv = self.gradientLoss2d(prediction)
        return loss_level + loss_tv

def reshape_tensor_to_2D(x):
    """
    reshape input variables of shape [B, C, D, H, W] to [voxel_n, C]
    """
    tensor_dim = len(x.size())
    num_class = list(x.size())[1]
    if (tensor_dim == 5):
        x_perm = x.permute(0, 2, 3, 4, 1)
    elif (tensor_dim == 4):
        x_perm = x.permute(0, 2, 3, 1)
    else:
        raise ValueError("{0:}D tensor not supported".format(tensor_dim))

    y = torch.reshape(x_perm, (-1, num_class))
    return y

def get_loss_and_confident_mask(pred, labels_prob, conf_ratio):
    prob = nn.Softmax(dim = 1)(pred)
    prob_2d = reshape_tensor_to_2D(prob) * 0.999 + 5e-4
    y_2d  = reshape_tensor_to_2D(labels_prob)
    loss = - y_2d * torch.log(prob_2d)
    loss = torch.sum(loss, dim = 1) # shape is [N]
    
    # loss = torch.mean(F.cross_entropy(prob, labels_prob, reduce = False), dim=[1,2])
    threshold = torch.quantile(loss, conf_ratio)    
    mask = loss < threshold
    
    return loss, mask

def loss_trinet_confidence_loss(y_1, y_2, y_3, t, remb_ratio, num_classes=2):
    # labels_prob = t.float()
    # labels_prob = torch.nn.functional.one_hot(t, num_classes).permute(0, 3, 1, 2)
    labels_prob = torch.nn.functional.one_hot(t, num_classes).permute(0, 4, 1, 2, 3)
    loss1, mask1 = get_loss_and_confident_mask(y_1, labels_prob, remb_ratio)
    loss2, mask2 = get_loss_and_confident_mask(y_2, labels_prob, remb_ratio)
    loss3, mask3 = get_loss_and_confident_mask(y_3, labels_prob, remb_ratio)
    mask12, mask13, mask23 = mask1 * mask2, mask1 * mask3, mask2 * mask3 
    mask12, mask13, mask23 = mask12.detach(), mask13.detach(), mask23.detach()
    
    loss1_avg = torch.sum(loss1 * mask23) / (mask23.sum() + 1e-6)
    loss2_avg = torch.sum(loss2 * mask13) / (mask13.sum() + 1e-6)
    loss3_avg = torch.sum(loss3 * mask12) / (mask12.sum() + 1e-6)
    loss = (loss1_avg + loss2_avg + loss3_avg) / 3
    
    return loss, loss1_avg, loss2_avg, loss3_avg


def loss_co_teaching(prob1, prob2, t, forget_ratio, num_classes=2):
    prob1_2d = reshape_tensor_to_2D(prob1) * 0.999 + 5e-4
    prob2_2d = reshape_tensor_to_2D(prob2) * 0.999 + 5e-4
    # labels_prob = torch.nn.functional.one_hot(t.to(torch.int64), num_classes).permute(0, 3, 1, 2)
    labels_prob = torch.nn.functional.one_hot(t, num_classes).permute(0, 4, 1, 2, 3)
    y_2d  = reshape_tensor_to_2D(labels_prob)

    loss1 = - y_2d* torch.log(prob1_2d)
    loss1 = torch.sum(loss1, dim = 1) # shape is [N]
    ind_1_sorted = torch.argsort(loss1)

    loss2 = - y_2d* torch.log(prob2_2d)
    loss2 = torch.sum(loss2, dim = 1) # shape is [N]
    ind_2_sorted = torch.argsort(loss2)

    
    remb_ratio   = 1 - forget_ratio
    num_remb = int(remb_ratio * len(loss1))

    ind_1_update = ind_1_sorted[:num_remb]
    ind_2_update = ind_2_sorted[:num_remb]

    loss1_select = loss1[ind_2_update]
    loss2_select = loss2[ind_1_update]
    
    loss = loss1_select.mean() + loss2_select.mean()
    return loss

class GeneralizedCELoss_pymic_copy(nn.Module):
    """
    Generalized cross entropy loss to deal with noisy labels. 

    * Reference: Z. Zhang et al. Generalized Cross Entropy Loss for Training Deep Neural Networks 
      with Noisy Labels, NeurIPS 2018.

    The parameters should be written in the `params` dictionary, and it has the following fields:

    :param `loss_softmax`: (bool) Apply softmax to the prediction of network or not.
    :param `loss_gce_q`: (float): hyper-parameter in the range of (0, 1).  
    :param `loss_with_pixel_weight`: (optional, bool): Use pixel weighting or not. 
    :param `loss_class_weight`: (optional, list or none): If not none, a list of weight for each class.
         
    """
    def __init__(self, gce_q):
        super().__init__()
        self.q = gce_q
        
    def forward(self, predict, target, softmax=False, num_classes=2):
        if(isinstance(predict, (list, tuple))):
            predict = predict[0]
        if softmax:
            predict = nn.Softmax(dim = 1)(predict)
        if len(target.shape) == 4:
            soft_y = torch.nn.functional.one_hot(target.to(torch.int64), num_classes).permute(0, 4, 1, 2, 3)
        elif len(target.shape) == 3:
            soft_y = torch.nn.functional.one_hot(target.to(torch.int64), num_classes).permute(0, 3, 1, 2)
            
        predict = reshape_tensor_to_2D(predict)
        soft_y  = reshape_tensor_to_2D(soft_y)
        gce     = (1.0 - torch.pow(predict, self.q)) / self.q * soft_y
        
        gce = torch.sum(gce, dim = 1)
        gce = torch.mean(gce)
        return gce

class NRDiceLoss(nn.Module):
    """
    Noise-robust Dice loss according to the following paper. 
        G. Wang et al. A Noise-Robust Framework for Automatic Segmentation of COVID-19 
        Pneumonia Lesions From CT Images, IEEE TMI, 2020.
    """
    def __init__(self, n_classes=2, p=1.5, softmax=True, weight=None):
        super(NRDiceLoss, self).__init__()
        self.n_classes = n_classes
        self.p = p
        self.softmax = softmax
        self.weight= weight

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _noise_robust_dice_loss(self, predict, soft_y):
        predict = torch.reshape(predict, (-1, 1))
        soft_y = torch.reshape(soft_y, (-1, 1))
        # numerator = torch.abs(predict - soft_y)
        # numerator = torch.pow(numerator,self.p)
        # numerator = torch.sum(numerator, dim=0)
        numerator = torch.sum(torch.pow(torch.abs(predict - soft_y), self.p), dim=0)
        y_vol = torch.sum(soft_y, dim=0)
        p_vol = torch.sum(predict, dim=0)
        # y_vol = torch.sum(torch.pow(soft_y, 2), dim=0)
        # p_vol = torch.sum(torch.pow(predict, 2), dim=0)
        loss = (numerator + 1e-5) / (y_vol + p_vol + 1e-5)
        # loss = torch.autograd.Variable(loss, requires_grad=True)
        # return torch.mean(loss)
        return loss

    def forward(self, inputs, target, num_classes=2):
        if self.softmax:
            inputs = torch.softmax(inputs, dim=1)      
        soft_y = torch.nn.functional.one_hot(target.to(torch.int64), num_classes).permute(0, 4, 1, 2, 3)
        # soft_y = target
        if self.weight is None:
            self.weight = [1] * self.n_classes
        assert inputs.size() == soft_y.size(), 'predict & target shape do not match'
        class_wise_NRDice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            NRDice = self._noise_robust_dice_loss(inputs[:, i], soft_y[:, i])
            class_wise_NRDice.append(NRDice.item())
            loss += NRDice * self.weight[i]
        return loss / self.n_classes


class GDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        """
        Generalized Dice;
        Copy from: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(GDiceLoss, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        shp_x = net_output.shape # (batch size,class_num,x,y,z)
        shp_y = gt.shape # (batch size,1,x,y,z)
        # one hot code for gt
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)


        if self.apply_nonlin is not None:
            softmax_output = self.apply_nonlin(net_output)
        else:
            softmax_output = net_output
        # copy from https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        w: torch.Tensor = 1 / (einsum("bcxyz->bc", y_onehot).type(torch.float32) + 1e-10)**2
        intersection: torch.Tensor = w * einsum("bcxyz, bcxyz->bc", softmax_output, y_onehot)
        union: torch.Tensor = w * (einsum("bcxyz->bc", softmax_output) + einsum("bcxyz->bc", y_onehot))
        divided: torch.Tensor = 1 - 2 * (einsum("bc->b", intersection) + self.smooth) / (einsum("bc->b", union) + self.smooth)
        gdc = divided.mean()

        return gdc