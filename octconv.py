import torch
import torch.nn as nn
import torch.nn.functional as F


def check_for_nan_inf(tensor, tensor_name):
    if tensor is None:
        return
    if torch.isnan(tensor).any():
        print(f"NaN detected in {tensor_name}")
    if torch.isinf(tensor).any():
        print(f"Inf detected in {tensor_name}")


class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=0, dilation=1,
                 groups=1, bias=False):
        super(OctaveConv, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        assert 0 <= alpha_in <= 1 and 0 <= alpha_out <= 1, "Alphas should be in the interval from 0 to 1."
        self.alpha_in, self.alpha_out = alpha_in, alpha_out
        self.conv_l2l = None if alpha_in == 0 or alpha_out == 0 else \
                        nn.Conv2d(int(alpha_in * in_channels), int(alpha_out * out_channels),
                                  kernel_size, 1, padding, dilation, groups, bias)
        self.conv_l2h = None if alpha_in == 0 or alpha_out == 1 else \
                        nn.Conv2d(int(alpha_in * in_channels), out_channels - int(alpha_out * out_channels),
                                  kernel_size, 1, padding, dilation, groups, bias)
        self.conv_h2l = None if alpha_in == 1 or alpha_out == 0 else \
                        nn.Conv2d(in_channels - int(alpha_in * in_channels), int(alpha_out * out_channels),
                                  kernel_size, 1, padding, dilation, groups, bias)
        self.conv_h2h = None if alpha_in == 1 or alpha_out == 1 else \
                        nn.Conv2d(in_channels - int(alpha_in * in_channels), out_channels - int(alpha_out * out_channels),
                                  kernel_size, 1, padding, dilation, groups, bias)

        self.pc = PartialConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        x_h, x_l = x if isinstance(x, tuple) else (x, None)
        check_for_nan_inf(x_h, "OctaveConv input x_h")
        check_for_nan_inf(x_l, "OctaveConv input x_l")

        if x_h is not None:
            x_h = self.downsample(x_h) if self.stride == 2 else x_h
            x_h2h = self.conv_h2h(self.pc(x_h))
            x_h2l = self.conv_h2l(self.downsample(self.pc(x_h))) if self.alpha_out > 0 else None
        if x_l is not None:
            x_l2h = self.conv_l2h(self.pc(x_l))
            x_l2h = self.upsample(x_l2h) if self.stride == 1 else x_l2h
            x_l2l = self.downsample(x_l) if self.stride == 2 else x_l
            x_l2l = self.conv_l2l(self.pc(x_l2l)) if self.alpha_out > 0 else None
            x_h2h = F.interpolate(x_h2h, (x_l2h.size()[2:]), mode='bilinear')
            x_h = x_l2h + x_h2h
            if x_h2l is not None:
                x_h2l = F.interpolate(x_h2l, (x_l2l.size()[2:]), mode='bilinear')
            x_l = x_h2l + x_l2l if x_h2l is not None and x_l2l is not None else None

        check_for_nan_inf(x_h, "OctaveConv output x_h")
        check_for_nan_inf(x_l, "OctaveConv output x_l")

        if x_l is not None:
            return x_h, x_l
        else:
            return x_h2h, x_h2l


class Conv_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(Conv_BN, self).__init__()
        self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha_in, alpha_out, stride, padding, dilation,
                               groups, bias)
        self.bn_h = None if alpha_out == 1 else norm_layer(int(out_channels * (1 - alpha_out)))
        self.bn_l = None if alpha_out == 0 else norm_layer(int(out_channels * alpha_out))

    def forward(self, x):
        x_h, x_l = self.conv(x)
        check_for_nan_inf(x_h, "Conv_BN input x_h")
        check_for_nan_inf(x_l, "Conv_BN input x_l")

        x_h = self.bn_h(x_h)
        x_l = self.bn_l(x_l) if x_l is not None else None

        check_for_nan_inf(x_h, "Conv_BN output x_h")
        check_for_nan_inf(x_l, "Conv_BN output x_l")
        return x_h, x_l


class Conv_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU):
        super(Conv_BN_ACT, self).__init__()
        self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha_in, alpha_out, stride, padding, dilation,
                               groups, bias)

        self.bn_h = None if alpha_out == 1 else norm_layer(int(out_channels * (1 - alpha_out)))
        self.bn_l = None if alpha_out == 0 else norm_layer(int(out_channels * alpha_out))
        self.act = activation_layer(inplace=False)
        self.incc = in_channels
        self.outcc = out_channels
        self.ali = alpha_in
        self.alo = alpha_out

    def forward(self, x):
        x_h, x_l = self.conv(x)
        check_for_nan_inf(x_h, "Conv_BN_ACT input x_h")
        check_for_nan_inf(x_l, "Conv_BN_ACT input x_l")

        x_h = self.act(self.bn_h(x_h))
        x_l = self.act(self.bn_l(x_l)) if x_l is not None else None

        check_for_nan_inf(x_h, "Conv_BN_ACT output x_h")
        check_for_nan_inf(x_l, "Conv_BN_ACT output x_l")
        return x_h, x_l


class OctMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__()
        self.maxpool_h = nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False,
                                      ceil_mode=False)
        self.maxpool_l = nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False,
                                      ceil_mode=False)

    def forward(self, x):
        h, l = x
        check_for_nan_inf(h, "OctMaxPool2d input h")
        check_for_nan_inf(l, "OctMaxPool2d input l")

        h_out = self.maxpool_h(h)
        l_out = self.maxpool_l(l)

        check_for_nan_inf(h_out, "OctMaxPool2d output h")
        check_for_nan_inf(l_out, "OctMaxPool2d output l")

        return h_out, l_out

class OctUp(nn.Module):
    def __init__(self, scale_factor=None, size=(None, None)):
        super().__init__()
        if scale_factor is not None:
            self.maxpool_h = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.maxpool_l = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.maxpool_h = nn.Upsample(size=size, mode='bilinear', align_corners=True)
            self.maxpool_l = nn.Upsample(size=size, mode='bilinear', align_corners=True)

    def forward(self, x):
        h, l = x
        check_for_nan_inf(h, "OctUp input h")
        check_for_nan_inf(l, "OctUp input l")

        h_out = self.maxpool_h(h)
        l_out = self.maxpool_l(l)

        check_for_nan_inf(h_out, "OctUp output h")
        check_for_nan_inf(l_out, "OctUp output l")

        return h_out, l_out


class OctUp_size(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        h, l = x1
        check_for_nan_inf(h, "OctUp_size input h")
        check_for_nan_inf(l, "OctUp_size input l")

        h_out = F.interpolate(h, x2[0].size()[2:], mode='bilinear')
        l_out = F.interpolate(l, x2[1].size()[2:], mode='bilinear')

        check_for_nan_inf(h_out, "OctUp_size output h")
        check_for_nan_inf(l_out, "OctUp_size output l")

        return h_out, l_out


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]

        self.last_size = (None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask=None):
        check_for_nan_inf(input, "PartialConv2d input")
        if mask is not None or self.last_size != (input.data.shape[2], input.data.shape[3]):
            self.last_size = (input.data.shape[2], input.data.shape[3])

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        if self.update_mask.type() != input.type() or self.mask_ratio.type() != input.type():
            self.update_mask = self.update_mask.to(input)
            self.mask_ratio = self.mask_ratio.to(input)
        if mask is not None:
            input = torch.mul(input, mask)

        check_for_nan_inf(input, "PartialConv2d output")
        return input
