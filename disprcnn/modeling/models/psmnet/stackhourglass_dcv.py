import math
import torch.utils.data
from disprcnn.modeling.psmnet.stackhourglass import hourglass
from .submodule import *


class PSMNet(nn.Module):
    def __init__(self, max_rel_depth, min_rel_depth, depth_interval):
        super(PSMNet, self).__init__()
        self.max_rel_depth = max_rel_depth
        self.min_rel_depth = min_rel_depth
        self.depth_interval = depth_interval
        self.feature_extraction = feature_extraction()

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

    def forward(self, inputs):
        """
        @param left:
        @param right:
        @param x1:
        @param x1p:
        @param x2:
        @param x2p:
        @param fuxbs:
        @return:
        """
        if isinstance(inputs, dict):
            left, right = inputs['left'], inputs['right']
            x1, x2, x1p, x2p, fuxbs = inputs['x1'].float(), inputs['x2'].float(), inputs[
                'x1p'].float(), inputs['x2p'].float(), inputs['fuxb'].float()
        else:
            left, right, x1, x2, x1p, x2p, fuxbs = inputs
        bsz, _, H, W = left.shape
        refimg_fea = self.feature_extraction(left)
        targetimg_fea = self.feature_extraction(right)
        device = refimg_fea.device
        _, C, Hp, Wp = refimg_fea.shape
        # matching
        cost_dimension = int((self.max_rel_depth - self.min_rel_depth) / self.depth_interval)
        cost = torch.zeros(bsz, C * 2,
                           cost_dimension
                           , Hp, Wp).float().to(device)
        crop_offsets = (x1 + x2) / 2 - (x1p + x2p) / 2
        crop_depths = fuxbs / crop_offsets  # coarse depth
        i = 0
        for dd in torch.arange(self.min_rel_depth, self.max_rel_depth, self.depth_interval):
            depth = crop_depths + dd
            disp = fuxbs / depth
            roi_disp = (disp - x1 + x1p) / torch.max(x2 - x1, x2p - x1p) * 224 / 4
            # compute grid
            is_zero = roi_disp == 0
            is_positive = roi_disp > 0
            is_negative = roi_disp < 0
            left_zero_cost = refimg_fea
            right_zero_cost = targetimg_fea
            xs = torch.arange(Wp).reshape(1, 1, Wp).repeat(bsz, Hp, 1).to(device).float()
            ys = torch.arange(Hp).reshape(1, Hp, 1).repeat(bsz, 1, Wp).to(device).float()
            base_grid = torch.stack((xs, ys), -1)
            grid = torch.stack((base_grid[:, :, :, 0] * 2 / Wp - 1,
                                base_grid[:, :, :, 1] * 2 / Hp - 1), -1)
            left_cost = F.grid_sample(refimg_fea, grid)
            pos_mask = (torch.arange(Wp).unsqueeze(0).repeat(bsz, 1).to(device).float() >= roi_disp[:, None]
                        ).int().unsqueeze(1).repeat(1, Hp, 1)
            neg_mask = (torch.arange(Wp).unsqueeze(0).repeat(bsz, 1).to(device).float() < roi_disp[:, None]
                        ).int().unsqueeze(1).repeat(1, Hp, 1)

            left_pos_cost = left_cost.clone() * pos_mask[:, None, :, :].float()
            left_neg_cost = left_cost.clone() * neg_mask[:, None, :, :].float()
            grid = torch.stack(((base_grid[:, :, :, 0] - roi_disp[:, None, None]) * 2 / Wp - 1,
                                base_grid[:, :, :, 1] * 2 / Hp - 1), -1)
            right_cost = F.grid_sample(targetimg_fea, grid)
            right_pos_cost = right_cost.clone() * pos_mask[:, None, :, :].float()
            right_neg_cost = right_cost.clone() * neg_mask[:, None, :, :].float()
            left_cost = is_zero[:, None, None, None].float() * left_zero_cost + \
                        is_positive[:, None, None, None].float() * left_pos_cost + \
                        is_negative[:, None, None, None].float() * left_neg_cost
            right_cost = is_zero[:, None, None, None].float() * right_zero_cost + \
                         is_positive[:, None, None, None].float() * right_pos_cost + \
                         is_negative[:, None, None, None].float() * right_neg_cost
            cost[:, :C, i] = left_cost
            cost[:, C:, i] = right_cost
            i += 1
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
            cost1 = F.interpolate(cost1, [cost_dimension, H, W], mode='trilinear',
                                  align_corners=True)
            cost2 = F.interpolate(cost2, [cost_dimension, H, W], mode='trilinear',
                                  align_corners=True)

            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparityregression(pred1, self.max_rel_depth, self.min_rel_depth,
                                        interval=self.depth_interval) + crop_depths[:, None, None]
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparityregression(pred2, self.max_rel_depth, self.min_rel_depth,
                                        interval=self.depth_interval) + crop_depths[:, None, None]
            cost3 = F.interpolate(cost3, [cost_dimension, H, W], mode='trilinear',
                                  align_corners=True)
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparityregression(pred3, self.max_rel_depth, self.min_rel_depth,
                                        interval=self.depth_interval) + crop_depths[:, None, None]
            return pred1, pred2, pred3
        else:
            cost3 = F.interpolate(cost3, [cost_dimension, H, W], mode='trilinear',
                                  align_corners=True)
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparityregression(pred3, self.max_rel_depth, self.min_rel_depth,
                                        interval=self.depth_interval) + crop_depths[:, None, None]
            return pred3
