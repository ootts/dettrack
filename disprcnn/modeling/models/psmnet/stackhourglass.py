import torch
import math

import torch.utils.data
from .submodule import *


class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes * 2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes))  # +x

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post


class PSMNet(nn.Module):
    def __init__(self, cfg, ):
        super(PSMNet, self).__init__()
        # maxdisp, mindisp = 0, input_size = 224, is_module = False,
        # feature_level = 1, single_modal_weight_average = False, conv_layers = (),
        # use_disparity_regression = True,
        # output_uncertainty = False,
        self.cfg = cfg.model.idispnet
        down = 7

        self.maxdisp = self.cfg.maxdisp
        self.mindisp = self.cfg.mindisp
        self.input_size = input_size = self.cfg.input_size

        # self.use_disparity_regression = use_disparity_regression
        # self.single_modal_weight_average = single_modal_weight_average
        # self.output_uncertainty = output_uncertainty
        # self.deconv = deconv
        # if not is_module:
        self.feature_extraction = feature_extraction(input_size, down)
        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        if self.cfg.pretrained_model != '':
            ckpt = torch.load(self.cfg.pretrained_model, 'cpu')
            if 'model' in ckpt:
                ckpt = ckpt['model']
            elif 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
                ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
            self.load_state_dict(ckpt)
        # print()

    def forward(self, inputs):
        # def forward(self, left, right):
        """
        @param left: Bx3xHxW
        @param right: Bx3xHxW
        # @param crop_x1s: Bx1
        # @param crop_x1ps: Bx1
        # @param fuxbs: Bx1
        @return:
        """
        # if isinstance(inputs, dict):
        left, right = inputs['left'], inputs['right']
        # elif len(inputs) == 2:
        #     left, right = inputs
        bsz, _, H, W = left.shape
        # if self.is_module:
        #     H = H * 4
        #     W = W * 4
        refimg_fea = self.feature_extraction(left)
        targetimg_fea = self.feature_extraction(right)

        _, C, Hp, Wp = refimg_fea.shape
        # matching
        cost = torch.zeros(bsz, C * 2, (self.maxdisp - self.mindisp) // 4, Hp, Wp).float().to(refimg_fea.device)
        # base_depths = fuxbs / (crop_x1s + crop_x1ps)
        # for dd in torch.arange(-8, 8, 0.1):
        #     depth = base_depths + dd
        #     disp = fuxbs / depth
        for i in range(self.mindisp // 4, self.maxdisp // 4):
            if i < 0:
                cost[:, :C, i - self.mindisp // 4, :, :i] = refimg_fea[:, :, :, :i]
                cost[:, C:, i - self.mindisp // 4, :, :i] = targetimg_fea[:, :, :, -i:]
            elif i > 0:
                cost[:, :C, i - self.mindisp // 4, :, i:] = refimg_fea[:, :, :, i:]
                cost[:, C:, i - self.mindisp // 4, :, i:] = targetimg_fea[:, :, :, :-i]
            else:
                cost[:, :C, i - self.mindisp // 4, :, :] = refimg_fea
                cost[:, C:, i - self.mindisp // 4, :, :] = targetimg_fea
        cost = cost.contiguous()

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3 + cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2
        if self.training:
            cost1 = F.interpolate(cost1, [self.maxdisp - self.mindisp, H, W], mode='trilinear',
                                  align_corners=True)
            cost2 = F.interpolate(cost2, [self.maxdisp - self.mindisp, H, W], mode='trilinear',
                                  align_corners=True)

            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            # if self.use_disparity_regression:
            pred1 = disparityregression(pred1, self.maxdisp, self.mindisp)

            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            # if self.use_disparity_regression:
            pred2 = disparityregression(pred2, self.maxdisp, self.mindisp)

            cost3 = F.interpolate(cost3, [self.maxdisp - self.mindisp, H, W], mode='trilinear',
                                  align_corners=True)
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            # For your information: This formulation 'softmax(c)' learned "similarity"
            # while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
            # However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
            # if self.use_disparity_regression:
            pred3 = disparityregression(pred3, self.maxdisp, self.mindisp)
            return pred1, pred2, pred3
        else:
            cost3 = F.interpolate(cost3, [self.maxdisp - self.mindisp, H, W], mode='trilinear',
                                  align_corners=True)
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            # if self.use_disparity_regression:
            p3 = disparityregression(pred3, self.maxdisp, self.mindisp)
            # if not self.output_uncertainty:
            return p3
            # else:
            #     uncert = torch.std(pred3, dim=1)
            #     return p3, uncert

    def forward_onnx(self, left, right):
        bsz, _, H, W = left.shape
        refimg_fea = self.feature_extraction(left)
        targetimg_fea = self.feature_extraction(right)
        _, C, Hp, Wp = refimg_fea.shape
        C, Hp, Wp = 32, 28, 28
        costs = []
        for i in range(-6, 6):
            if i < 0:
                tmp1 = torch.cat([refimg_fea[:, :, :, :Wp + i],
                                  torch.zeros([1, 32, 28, -i]).cuda().float()], dim=-1)
                tmp2 = torch.cat([targetimg_fea[:, :, :, -i:],
                                  torch.zeros([1, 32, 28, -i]).cuda().float()], dim=-1)
                costs.append(torch.cat([tmp1, tmp2], dim=1))
            elif i > 0:
                tmp1 = torch.cat([torch.zeros([1, 32, 28, i]).cuda().float(),
                                  refimg_fea[:, :, :, i:]], dim=-1)
                tmp2 = torch.cat([torch.zeros([1, 32, 28, i]).cuda().float(),
                                  targetimg_fea[:, :, :, :Wp - i]], dim=-1)
                costs.append(torch.cat([tmp1, tmp2], dim=1))
            else:
                costs.append(torch.cat([refimg_fea, targetimg_fea], dim=1))
        cost = torch.stack(costs, dim=2)
        cost = cost.contiguous()
        # return cost
        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3 + cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2
        assert not self.training
        cost3 = F.interpolate(cost3, [48, H, W], mode='trilinear', align_corners=True)
        cost3 = torch.squeeze(cost3, 1)
        pred3 = F.softmax(cost3, dim=1)
        p3 = disparityregression(pred3, self.maxdisp, self.mindisp)
        return p3
