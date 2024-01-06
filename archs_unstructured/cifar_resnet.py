from __future__ import absolute_import

'''
This file is from: https://raw.githubusercontent.com/bearpaw/pytorch-classification/master/models/cifar/resnet.py
by Wei Yang
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init
from torch.nn import ConstantPad2d
from .init_utils import weights_init

from torch.autograd import Variable


__all__ = ['ResNet', 'resnet34_in', 'resnet50_in', 'resnet18_in', 'resnet9_in',
           'wide_resnet22_8', 'wide_resnet_22_8_drop02', 'wide_resnet_28_10_drop02', 'wide_resnet_28_12_drop02', 'wide_resnet_16_8_drop02'
           'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 
           'vgg19_bn', 'vgg19', 'lenet_5_caffe']


def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

def drop_path_(x, drop_prob, training):
    if training and drop_prob > 0.:
        keep_prob = 1. - drop_prob
        # per data point mask; assuming x in cuda.
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob).mul_(mask)

    return x


class DropPath_(nn.Module):
    def __init__(self, p=0.):
        """ [!] DropPath is inplace module
        Args:
            p: probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def extra_repr(self):
        return 'p={}, inplace'.format(self.p)

    def forward(self, x):
        drop_path_(x, self.p, self.training)

        return x


class LearnableAlpha(nn.Module):
    def __init__(self, out_channel, feature_size):
        super(LearnableAlpha, self).__init__()
        self.alphas = nn.Parameter(torch.ones(1, out_channel, feature_size, feature_size), requires_grad=True)

    def forward(self, x):
        out = F.relu(x) * self.alphas.expand_as(x) + (1-self.alphas.expand_as(x)) * x 
        return out


class LearnableAlphaAndBetaNewAlgorithm(nn.Module):
    def __init__(self, out_channel, feature_size):
        super(LearnableAlphaAndBetaNewAlgorithm, self).__init__()
        self.out_channel = out_channel
        self.feature_size = feature_size

        self.alphas = nn.Parameter(torch.ones(1, out_channel, feature_size, feature_size), requires_grad=True)
        self.beta = nn.Parameter(torch.cat(
            [
                # betas are now linear
                torch.eye(feature_size * feature_size, feature_size * feature_size).unsqueeze(0) * 1
            ] * out_channel, 0),
            requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1, out_channel, feature_size, feature_size), requires_grad=True)

    def set_default_params_beta_and_gamma(self):
        feature_size = self.feature_size
        out_channel = self.out_channel
        self.gamma.data = 1 - self.alphas.data
        self.beta.data = torch.cat(
            [
                # betas are now linear
                torch.eye(feature_size * feature_size, feature_size * feature_size).unsqueeze(0) * 1
            ] * out_channel, 0).to(self.gamma.device)

    def forward(self, x):
        B, C = x.shape[0], x.shape[1]
        drelu_x = (x > 0).float()
        per_channel_beta_times_alpha_times_drelu_and_sum = []
        for channel in range(C):
            # we want to repeat alpha to be the shape of beta (duplicate it over the rows)
            alpha_expanded = self.alphas[:, channel, :, :].flatten(-2).expand_as(self.beta[channel])
            # then we want to calculate the elementwise product to achieve b_{ij} * a_{j}
            beta_times_alpha = self.beta[channel] * alpha_expanded
            # finally we want to evaluate the sum of the elemetwise product with d_{j}, that is:
            # sum( b_{ij} * a_{j} * d_{j} ) * x_{j})
            beta_times_alpha_times_drelu = (beta_times_alpha @ drelu_x[..., channel, :, :].flatten(
                -2).T).T  # / self.feature_size / self.feature_size
            # then we want to reshape this result and add it to the list...
            beta_times_alpha_times_drelu_as_x_shape = beta_times_alpha_times_drelu.reshape(B, 1, self.feature_size,
                                                                                           self.feature_size)
            per_channel_beta_times_alpha_times_drelu_and_sum.append(beta_times_alpha_times_drelu_as_x_shape)
        new_drelu = torch.cat(per_channel_beta_times_alpha_times_drelu_and_sum, 1)
        new_drelu = new_drelu + self.gamma.expand_as(x)
        new_relu = new_drelu * x
        out = new_relu  # + 0.05 * x
        return out


class LearnableAlphaBetaGamma(nn.Module):
    def __init__(self, out_channel, feature_size):
        super(LearnableAlphaBetaGamma, self).__init__()
        self.out_channel = out_channel
        self.feature_size = feature_size

        self.alphas = nn.Parameter(torch.ones(1, out_channel, feature_size, feature_size), requires_grad=True)
        self.beta = nn.Parameter(torch.cat(
            [
                # betas are now linear
                torch.eye(feature_size * feature_size, feature_size * feature_size).unsqueeze(0) * 1
            ] * out_channel, 0),
            requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1, out_channel, feature_size, feature_size), requires_grad=True)
        self.threshold = 1e-2

    def set_default_params_beta_and_gamma(self):
        feature_size = self.feature_size
        out_channel = self.out_channel
        self.gamma.data = 1 - self.alphas.data
        self.beta.data = torch.cat(
            [
                # betas are now linear
                torch.eye(feature_size * feature_size, feature_size * feature_size).unsqueeze(0) * 1
            ] * out_channel, 0).to(self.gamma.device)


    def forward(self, x):
        B, C = x.shape[0], x.shape[1]
        drelu_x = (x > 0).float()
        per_channel_beta_times_alpha_times_drelu_and_sum = []
        for channel in range(C):
            # we want to repeat alpha to be the shape of beta (duplicate it over the rows)
            alpha_expanded = self.alphas[:, channel, :, :].flatten(-2).expand_as(self.beta[channel])
            # then we want to calculate the elementwise product to achieve b_{ij} * a_{j}
            beta_times_alpha = self.beta[channel] * alpha_expanded
            # finally we want to evaluate the sum of the elemetwise product with d_{j}, that is:
            # sum( b_{ij} * a_{j} * d_{j} ) * x_{j})
            beta_times_alpha_times_drelu = (beta_times_alpha @ drelu_x[..., channel, :, :].flatten(
                -2).T).T  # / self.feature_size / self.feature_size
            # then we want to reshape this result and add it to the list...
            beta_times_alpha_times_drelu_as_x_shape = beta_times_alpha_times_drelu.reshape(B, 1, self.feature_size,
                                                                                           self.feature_size)
            per_channel_beta_times_alpha_times_drelu_and_sum.append(beta_times_alpha_times_drelu_as_x_shape)
        new_drelu = torch.cat(per_channel_beta_times_alpha_times_drelu_and_sum, 1)
        new_drelu = new_drelu + self.gamma.expand_as(x)
        new_drelu = new_drelu * (self.alphas.expand_as(x) < self.threshold).float() + (
                self.alphas.expand_as(x) >= self.threshold).float() * (drelu_x* self.alphas.expand_as(x) + (1-self.alphas.expand_as(x)) * 1)
        new_relu = new_drelu * x
        out = new_relu  # + 0.05 * x
        return out


class LearnableAlphaAndBeta(nn.Module):
    def __init__(self, out_channel, feature_size):
        super(LearnableAlphaAndBeta, self).__init__()
        self.alphas = nn.Parameter(torch.ones(1, out_channel, feature_size, feature_size), requires_grad=True)
        self.beta = nn.Parameter(torch.cat(
            [
                # betas are at the argument of a sigmoid, therefore we need to initialize them with A and -A.
                # This A should be large enough if we want to get 1 and 0. Otherwise, since we sum over H*W elements,
                # small fractions close to 0 add up.
                torch.eye(feature_size * feature_size, feature_size * feature_size).unsqueeze(0) * 30 - 15
            ] * out_channel, 0),
            requires_grad=True)
        self.feature_size = feature_size

    def alternative_forward(self, x):

        B, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

        drelu_x = (x > 0).float()

        alphas_times_drelu = (drelu_x * self.alphas.expand_as(drelu_x)).unsqueeze(-1)
        betas_expanded = torch.nn.Sigmoid()(self.beta.broadcast_to((B, C, H * W, H * W)))

        estimated_drelu = torch.matmul(betas_expanded, alphas_times_drelu).squeeze(-1).reshape(B, C, H, W)

        return estimated_drelu * x


    def forward(self, x):
        B, C = x.shape[0], x.shape[1]
        drelu_x = (x > 0).float()
        per_channel_beta_times_alpha_times_drelu_and_sum = []
        for channel in range(C):
            # we want to repeat alpha to be the shape of beta (duplicate it over the rows)
            alpha_expanded = self.alphas[:, channel, :, :].flatten(-2).expand_as(self.beta[channel])
            # then we want to calculate the elementwise product to achieve b_{ij} * a_{j}
            beta_times_alpha = torch.nn.Sigmoid()(self.beta[channel]) * alpha_expanded
            # finally we want to evaluate the sum of the elemetwise product with d_{j}, that is:
            # sum( b_{ij} * a_{j} * d_{j} ) * x_{j})
            beta_times_alpha_times_drelu = (beta_times_alpha @ drelu_x[..., channel, :, :].flatten(-2).T).T # / self.feature_size / self.feature_size
            # then we want to reshape this result and add it to the list...
            beta_times_alpha_times_drelu_as_x_shape = beta_times_alpha_times_drelu.reshape(B, 1, self.feature_size, self.feature_size)
            per_channel_beta_times_alpha_times_drelu_and_sum.append(beta_times_alpha_times_drelu_as_x_shape)
        new_drelu = torch.cat(per_channel_beta_times_alpha_times_drelu_and_sum, 1)
        new_relu = new_drelu * x
        out = new_relu #+ 0.05 * x
        return out


class LearnableAlphaAndBetaNoSigmoid(nn.Module):
    def __init__(self, out_channel, feature_size):
        super(LearnableAlphaAndBetaNoSigmoid, self).__init__()
        self.alphas = nn.Parameter(torch.ones(1, out_channel, feature_size, feature_size), requires_grad=True)
        self.beta = nn.Parameter(torch.cat(
            [
                # betas are now linear
                torch.eye(feature_size * feature_size, feature_size * feature_size).unsqueeze(0) * 1
            ] * out_channel, 0),
            requires_grad=True)
        self.feature_size = feature_size

    def forward(self, x):
        B, C = x.shape[0], x.shape[1]
        drelu_x = (x > 0).float()
        per_channel_beta_times_alpha_times_drelu_and_sum = []
        for channel in range(C):
            # we want to repeat alpha to be the shape of beta (duplicate it over the rows)
            alpha_expanded = self.alphas[:, channel, :, :].flatten(-2).expand_as(self.beta[channel])
            # then we want to calculate the elementwise product to achieve b_{ij} * a_{j}
            beta_times_alpha = self.beta[channel] * alpha_expanded
            # finally we want to evaluate the sum of the elemetwise product with d_{j}, that is:
            # sum( b_{ij} * a_{j} * d_{j} ) * x_{j})
            beta_times_alpha_times_drelu = (beta_times_alpha @ drelu_x[..., channel, :, :].flatten(-2).T).T  # / self.feature_size / self.feature_size
            # then we want to reshape this result and add it to the list...
            beta_times_alpha_times_drelu_as_x_shape = beta_times_alpha_times_drelu.reshape(B, 1, self.feature_size,
                                                                                           self.feature_size)
            per_channel_beta_times_alpha_times_drelu_and_sum.append(beta_times_alpha_times_drelu_as_x_shape)
        new_drelu = torch.cat(per_channel_beta_times_alpha_times_drelu_and_sum, 1)
        new_relu = new_drelu * x
        out = new_relu # + 0.05 * x
        return out



class LearnableAlphaWithEpsilon(nn.Module):
    def __init__(self, out_channel, feature_size, epsilon, num_of_neighbors: int = 4):
        super(LearnableAlphaWithEpsilon, self).__init__()
        self.epsilon = epsilon
        self.alphas = nn.Parameter(torch.ones(1, out_channel, feature_size, feature_size) * 0.5, requires_grad=True)
        self.product_function = {4: self.calculate_product_for_four_neighbors,
                                 8: self.calculate_product_for_eight_neighbors}[num_of_neighbors]

    @staticmethod
    def calculate_product_for_four_neighbors(our_drelu, alphas):
        """Calculate the product of the DReLUs of a pixel and its four spatially nearest neighbors.

        We follow the following convention for the naming of the pixels surrounding the pixel X:
            ╔═══════════╗
            ║ A ║ B ║ C ║
            ╟───────────╢
            ║ D ║ X ║ E ║
            ╟───────────╢
            ║ F ║ G ║ H ║
            ╚═══════════╝
        We calculate the following equation:
        \prod_j (1-\alpha_j) \cdot \text{Our-DReLU}(x_j)


        """

        padding_left, padding_right, padding_top, padding_bottom = (0, 0, 1, 0)
        B = (ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 1)(our_drelu)[..., :-1, :] *
             ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 1)(
                 1.0 - alphas)[..., :-1, :])

        padding_left, padding_right, padding_top, padding_bottom = (1, 0, 0, 0)
        D = (ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 1)(our_drelu)[..., :, :-1] *
             ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 1)(
                 1.0 - alphas)[..., :, :-1])
        X = our_drelu * (1.0 - alphas)
        padding_left, padding_right, padding_top, padding_bottom = (0, 1, 0, 0)
        E = (ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 1)(our_drelu)[..., :, 1:] *
             ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 1)(
                 1.0 - alphas)[..., :, 1:])

        padding_left, padding_right, padding_top, padding_bottom = (0, 0, 0, 1)
        G = (ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 1)(our_drelu)[..., 1:, :] *
             ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 1)(
                 1.0 - alphas)[..., 1:, :])

        product = B * D * X * G
        return product

    @staticmethod
    def calculate_product_for_eight_neighbors(our_drelu, alphas):
        """Calculate the product of the DReLUs of a pixel and its eight spatially nearest neighbors.

        We follow the following convention for the naming of the pixels surrounding the pixel X:
            ╔═══════════╗
            ║ A ║ B ║ C ║
            ╟───────────╢
            ║ D ║ X ║ E ║
            ╟───────────╢
            ║ F ║ G ║ H ║
            ╚═══════════╝
        We calculate the following equation:
        \prod_j (1-\alpha_j) \cdot \text{Our-DReLU}(x_j)

                """
        padding_left, padding_right, padding_top, padding_bottom = (1, 0, 1, 0)
        A = (ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 1)(our_drelu)[..., :-1 , :-1] *
             ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 1)(
                 1.0 -  alphas)[..., :-1 , :-1])
        padding_left, padding_right, padding_top, padding_bottom = (0, 0, 1, 0)
        B = (ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 1)(our_drelu)[..., :-1, :] *
             ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 1)(
                 1.0 - alphas)[..., :-1, :])
        padding_left, padding_right, padding_top, padding_bottom = (0, 1, 1, 0)
        C = (ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 1)(our_drelu)[..., :-1 , 1:] *
             ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 1)(
                 1.0 - alphas)[..., :-1, 1:])
        padding_left, padding_right, padding_top, padding_bottom = (1, 0, 0, 0)
        D = (ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 1)(our_drelu)[..., :, :-1] *
             ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 1)(
                 1.0 - alphas)[..., :, :-1])
        X = our_drelu * (1.0 - alphas)
        padding_left, padding_right, padding_top, padding_bottom = (0, 1, 0, 0)
        E = (ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 1)(our_drelu)[..., :, 1:] *
             ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 1)(
                 1.0 - alphas)[..., :, 1:])
        padding_left, padding_right, padding_top, padding_bottom = (1, 0, 0, 1)
        F = (ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 1)(our_drelu)[..., :-1, 1:] *
             ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 1)(
                 1.0 - alphas)[..., :-1, 1:])
        padding_left, padding_right, padding_top, padding_bottom = (0, 0, 0, 1)
        G = (ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 1)(our_drelu)[..., 1:, :] *
             ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 1)(
                 1.0 - alphas)[..., 1:, :])
        padding_left, padding_right, padding_top, padding_bottom = (0, 1, 0, 1)
        H = (ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 1)(our_drelu)[..., 1:, 1:] *
             ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 1)(
                 1.0 - alphas)[..., 1:, 1:])
        product = A * B * C * D * X * E * F * G * H
        return product

    def forward(self, x):
        which_drelus_we_calculate = (self.alphas.expand_as(x) > self.epsilon).float()
        our_drelu = which_drelus_we_calculate * (x > 0).float() + 1 * (1.0 -  which_drelus_we_calculate)
        product = self.product_function(our_drelu, self.alphas.expand_as(x))
        out = product * x
        return out



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.alpha1 = LearnableAlpha(planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.alpha2 = LearnableAlpha(planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn1(self.conv1(x))
        out = self.alpha1(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        # out = F.relu(out)
        out = self.alpha2(out)
        return out

class BasicBlock_IN(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, feature_size, args):
        super(BasicBlock_IN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.alpha1 = {'LearnableAlpha': LearnableAlpha(planes, feature_size),
                       'LearnableAlphaWithEpsilon': LearnableAlphaWithEpsilon(planes, feature_size, epsilon=args.threshold, num_of_neighbors=args.num_of_neighbors),
                       'LearnableAlphaAndBeta': LearnableAlphaAndBeta(planes, feature_size),
                       'LearnableAlphaAndBetaNoSigmoid': LearnableAlphaAndBetaNoSigmoid(planes, feature_size),
                       'LearnableAlphaAndBetaNewAlgorithm': LearnableAlphaAndBetaNewAlgorithm(planes, feature_size),
                       'LearnableAlphaBetaGamma': LearnableAlphaBetaGamma(planes, feature_size),
                       }[args.block_type]
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.alpha2 = {'LearnableAlpha': LearnableAlpha(planes, feature_size),
                       'LearnableAlphaWithEpsilon': LearnableAlphaWithEpsilon(planes, feature_size, epsilon=args.threshold, num_of_neighbors=args.num_of_neighbors),
                       'LearnableAlphaAndBeta': LearnableAlphaAndBeta(planes, feature_size),
                       'LearnableAlphaAndBetaNoSigmoid': LearnableAlphaAndBetaNoSigmoid(planes, feature_size),
                       'LearnableAlphaAndBetaNewAlgorithm': LearnableAlphaAndBetaNewAlgorithm(planes, feature_size),
                       'LearnableAlphaBetaGamma': LearnableAlphaBetaGamma(planes, feature_size),
                       }[args.block_type]

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn1(self.conv1(x))
        out = self.alpha1(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        # out = F.relu(out)
        out = self.alpha2(out)
        return out



class ResNet_IN(nn.Module):
    def __init__(self, block, num_blocks, args, num_classes=10):
        super(ResNet_IN, self).__init__()
        self.in_planes = 64
        if args.dataset in ['cifar10', 'cifar100', "cifar100-new-split"]:
            self.feature_size = 32
            self.last_dim = 4
            print("CIFAR10/100 Setting")
        elif args.dataset in ['tiny_imagenet']:
            self.feature_size = 64
            self.last_dim = 8
            print("Tiny_ImageNet Setting")
            print("num_classes: ", num_classes)
        else:
            raise ValueError("Dataset not implemented for ResNet_IN")
        self.args = args
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=args.stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.alpha = LearnableAlpha(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, args=args)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, args=args)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, args=args)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, args=args)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.apply(_weights_init)

        
    def _make_layer(self, block, planes, num_blocks, stride, args):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            self.feature_size = self.feature_size // 2 if stride == 2 else self.feature_size
            layers.append(block(self.in_planes, planes, stride, self.feature_size, args=args))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, self.last_dim if self.args.stride == 1 else self.last_dim // 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

def resnet9_in(num_classes, args):
    return ResNet_IN(BasicBlock_IN, [1, 1, 1, 1], num_classes=num_classes, args=args)

def resnet18_in(num_classes, args):
    return ResNet_IN(BasicBlock_IN, [2, 2, 2, 2], num_classes=num_classes, args=args)

def resnet34_in(num_classes, args):
    return ResNet_IN(BasicBlock_IN, [3, 4, 6, 3], num_classes=num_classes, args=args)

def resnet50_in(num_classes, args):
    return ResNet_IN(BasicBlock_IN, [3, 4, 14, 3], num_classes=num_classes, args=args)
    

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, feature_size, dropRate=0.0):
        super(WideBasicBlock, self).__init__()
        self.equalInOut = (in_planes == out_planes)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        # self.alpha1 = LearnableAlpha(in_planes, feature_size)
        if not self.equalInOut and stride != 1:
            self.alpha1 = LearnableAlpha(in_planes, 2*feature_size)
        else:
            self.alpha1 = LearnableAlpha(in_planes, feature_size)
        
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.alpha2 = LearnableAlpha(out_planes, feature_size)
        self.droprate = dropRate
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            # x = self.relu1(self.bn1(x))
            x = self.alpha1(self.bn1(x))
        else:
            # out = self.relu1(self.bn1(x))
            out = self.alpha1(self.bn1(x))
        # out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        out = self.alpha2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, feature_size, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, feature_size, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, feature_size, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, feature_size, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, args, depth=22, num_classes=10, widen_factor=8, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        if args.dataset in ['cifar10', 'cifar100']:
            self.feature_size = 32
            self.last_dim = 8
        elif args.dataset in ['tiny_imagenet']:
            self.feature_size = 64
            self.last_dim = 16
        else:
            raise ValueError("Dataset not implemented in WideResNet")
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = WideBasicBlock
        self.args = args
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=args.stride,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, self.feature_size, dropRate)
        # 2nd block
        self.feature_size = self.feature_size // 2
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, self.feature_size, dropRate)
        # 3rd block
        self.feature_size = self.feature_size // 2
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, self.feature_size, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        # self.alpha = LearnableAlpha(nChannels[3])
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.bn1(out)
        out = F.avg_pool2d(out, self.last_dim if self.args.stride == 1 else self.last_dim // 2)
        out = self.fc(out.view(-1, self.nChannels))
        return out
    
def wide_resnet_16_8_drop02(num_classes, args):
    return WideResNet(depth=16, num_classes=num_classes, widen_factor=8, dropRate=0.2, args=args)

def wide_resnet_22_8(num_classes, args):
    return WideResNet(depth=22, num_classes=num_classes, widen_factor=8, args=args)

def wide_resnet_22_8_drop02(num_classes, args):
    return WideResNet(depth=22, num_classes=num_classes, widen_factor=8, dropRate=0.2, args=args)

def wide_resnet_28_10_drop02(num_classes, args):
    return WideResNet(depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.2, args=args)

def wide_resnet_28_12_drop02(num_classes, args):
    return WideResNet(depth=28, num_classes=num_classes, widen_factor=12, dropRate=0.2, args=args)

def wide_resnet(**kwargs):
    return WideResNet(**kwargs)


defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class VGG(nn.Module):
    def __init__(self, num_classes=10, depth=19, init_weights=True, cfg=None, affine=True, batchnorm=True):
        super(VGG, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]
        self._AFFINE = affine
        self.feature = self.make_layers(cfg, batchnorm)
        self.num_classes = num_classes
        
        self.classifier = nn.Linear(cfg[-1], num_classes)
        if init_weights:
            self.apply(weights_init)
        # if pretrained:
        #     model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v, affine=self._AFFINE), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        if self.num_classes == 200:
            x = nn.AvgPool2d(4)(x)
        else:
            x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        y = F.log_softmax(x, dim=1)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1.0)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def vgg19(num_classes):
    """VGG 19-layer model (configuration "E")"""
    return VGG(num_classes)