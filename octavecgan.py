import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from octaveunet import OctaveUNet
from torchvision.models import resnet18

# cGAN discriminator architecture and utilities are adopted from https://github.com/mrzhu-cool/pix2pix-pytorch


import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from config import get_args

# Parse arguments
args = get_args()

# Set up device
if args.device == 'TPU':
    import torch_xla
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
else:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'



# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1 or classname.find('InstanceNorm2d') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname == 'OctaveConv':
        if m.conv_l2l is not None:
            nn.init.normal_(m.conv_l2l.weight.data, 0.0, 0.02)
        if m.conv_l2h is not None:
            nn.init.normal_(m.conv_l2h.weight.data, 0.0, 0.02)
        if m.conv_h2l is not None:
            nn.init.normal_(m.conv_h2l.weight.data, 0.0, 0.02)
        if m.conv_h2h is not None:
            nn.init.normal_(m.conv_h2h.weight.data, 0.0, 0.02)

def get_norm_layer(norm_type):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = nn.InstanceNorm2d
    else:
        print('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc=1, output_nc=1, norm='batch', use_dropout=False, gpu_ids=[]):
    netG = OctaveUNet(n_classes=output_nc, in_ch=input_nc)
    netG.apply(weights_init)  # مقداردهی اولیه وزن‌ها
    netG = netG.to(device)  # انتقال به TPU
    return netG

def define_D(input_nc=1, ndf=64, norm='batch', use_sigmoid=True, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    netD.apply(weights_init)  # مقداردهی اولیه وزن‌ها
    netD = netD.to(device)  # انتقال به TPU
    return netD

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None

        self.Tensor = tensor
        
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label).to(device)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label).to(device)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        input = input.to(device)
        target_tensor = self.get_target_tensor(input, target_is_real)
            
        return self.loss(input, target_tensor)


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult, affine=True),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult, affine=True),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        # sequence += [nn.AdaptiveAvgPool2d(1)]  # اضافه کtarget_is_realردن AdaptiveAvgPool2d به جای Flatten
        # sequence += [nn.Flatten()]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
