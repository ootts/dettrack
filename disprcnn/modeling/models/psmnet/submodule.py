from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Conv2d, SyncBatchNorm, BatchNorm2d, BatchNorm3d

B2d = BatchNorm2d
B3d = BatchNorm3d


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         B2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
        B3d(out_planes))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class matchshifted(nn.Module):
    def __init__(self):
        super(matchshifted, self).__init__()

    def forward(self, left, right, shift):
        batch, filters, height, width = left.size()
        shifted_left = F.pad(
            torch.index_select(left, 3, Variable(torch.LongTensor([i for i in range(shift, width)])).cuda()),
            (shift, 0, 0, 0))
        shifted_right = F.pad(
            torch.index_select(right, 3, Variable(torch.LongTensor([i for i in range(width - shift)])).cuda()),
            (shift, 0, 0, 0))
        out = torch.cat((shifted_left, shifted_right), 1).view(batch, filters * 2, 1, height, width)
        return out


def single_modal_weight_average_v4(x: torch.Tensor):  # 1min 41s
    """
    :param x: bsz, disp_range, H, W
    :return:
    """
    bsz, disp_range, H, W = x.shape
    keep = torch.zeros_like(x).byte()
    maxidx = x.argmax(dim=1)
    keep.scatter_(1, maxidx.unsqueeze(1), 1)
    l = (maxidx.clone() - 1).clamp(min=0)
    go = torch.ones((bsz, H, W)).int().to(x.device)
    while l.sum() != 0:
        to_write = x.gather(1, l.clamp(min=0).unsqueeze(1)) < x.gather(1, (l + 1).unsqueeze(1))
        go = ((go.int()) & (to_write.int())).byte()
        if torch.all(go == 0):
            # break when all pixels have reached local minimum
            break
        keep = keep | keep.scatter(1, l.unsqueeze(1), to_write * go)
        l = torch.clamp(l - 1, min=0)
    r = (maxidx.clone() + 1).clamp(max=disp_range - 1)
    go = torch.ones((bsz, H, W)).int().to(x.device)
    while not torch.all(r == disp_range - 1):
        to_write = x.gather(1, r.clamp(max=disp_range - 1).unsqueeze(1)) < x.gather(1, (r - 1).unsqueeze(1))
        go = ((go.int()) & (to_write.int())).byte()
        if torch.all(go == 0):
            # break when all pixels have reached local minimum
            break
        keep = keep | keep.scatter(1, r.unsqueeze(1), to_write * go)
        r = torch.clamp(r + 1, max=disp_range - 1)
    xp = x * keep.float()
    xp = xp / xp.sum(dim=1, keepdim=True)
    assert torch.isnan(xp).sum() == 0
    return xp


def disparityregression(x, maxdisp, mindisp=0, single_modal_weight_average=False, interval=1):
    assert x.shape[1] == int((maxdisp - mindisp) / interval)
    bsz, disp_range, H, W = x.shape
    disp = torch.arange(mindisp, maxdisp, interval).reshape((1, disp_range, 1, 1)).float().to(x.device)
    disp = disp.repeat((bsz, 1, H, W))
    if single_modal_weight_average:
        x = single_modal_weight_average_v4(x)
    out = torch.sum(x * disp, 1)
    return out


class feature_extraction(nn.Module):
    def __init__(self, input_size=224, down=7):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)
        apsize = input_size // 4
        self.branch1 = nn.Sequential(
            nn.AvgPool2d((apsize, apsize), stride=(apsize, apsize)),
            # nn.AdaptiveAvgPool2d((1, 1)),
            convbn(128, 32, 1, 1, 0, 1),
            nn.ReLU(inplace=True))
        if down == 7:
            apsize = input_size // 7
        else:
            apsize = input_size // 8
        self.branch2 = nn.Sequential(
            nn.AvgPool2d((apsize, apsize), stride=(apsize, apsize)),
            # nn.AdaptiveAvgPool2d((1, 1)),
            convbn(128, 32, 1, 1, 0, 1),
            nn.ReLU(inplace=True))
        if down == 7:
            apsize = input_size // 14
        else:
            apsize = input_size // 16
        self.branch3 = nn.Sequential(
            nn.AvgPool2d((apsize, apsize), stride=(apsize, apsize)),
            # nn.AdaptiveAvgPool2d((1, 1)),
            convbn(128, 32, 1, 1, 0, 1),
            nn.ReLU(inplace=True))
        if down == 7:
            apsize = input_size // 28
        else:
            apsize = input_size // 32
        self.branch4 = nn.Sequential(
            nn.AvgPool2d((apsize, apsize), stride=(apsize, apsize)),
            # nn.AdaptiveAvgPool2d((1, 1)),
            convbn(128, 32, 1, 1, 0, 1),
            nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2d(self.inplanes, planes * block.expansion,
                       kernel_size=1, stride=stride, bias=False),
                B2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.firstconv(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)

        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.interpolate(output_branch1, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear',
                                       align_corners=True)

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.interpolate(output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear',
                                       align_corners=True)

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.interpolate(output_branch3, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear',
                                       align_corners=True)

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.interpolate(output_branch4, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear',
                                       align_corners=True)

        output_feature = torch.cat(
            (output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature
