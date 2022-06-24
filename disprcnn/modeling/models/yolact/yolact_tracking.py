import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from dl_ext.timer import EvalTime
from torchvision.ops import RoIAlign

from disprcnn.structures.bounding_box import BoxList

from disprcnn.modeling.models.yolact.submodules import *
from disprcnn.structures.boxlist_ops import boxlist_iou
from .yolact import Yolact


class TrackHead(nn.Module):
    def __init__(self, cfg):
        super(TrackHead, self).__init__()
        self.total_cfg = cfg
        self.cfg = cfg.model.yolact_tracking.track_head
        in_channels = self.cfg.in_channels
        self.in_channels = in_channels
        self.roi_feat_size = self.cfg.roi_feat_size
        self.match_coeff = self.cfg.match_coeff
        self.bbox_dummy_iou = self.cfg.bbox_dummy_iou
        num_fcs = self.cfg.num_fcs
        fc_out_channels = self.cfg.fc_out_channels
        dynamic = self.cfg.dynamic

        in_channels *= (self.roi_feat_size * self.roi_feat_size)
        self.fcs = nn.ModuleList()
        for i in range(num_fcs):
            in_channels = (in_channels
                           if i == 0 else fc_out_channels)
            fc = nn.Linear(in_channels, fc_out_channels)
            self.fcs.append(fc)
        self.relu = nn.ReLU(inplace=True)
        self.dynamic = dynamic

        self.init_weights()

    def init_weights(self):
        for fc in self.fcs:
            nn.init.normal_(fc.weight, 0, 0.01)
            nn.init.constant_(fc.bias, 0)

    def compute_comp_scores(self, match_ll, bbox_scores, bbox_ious, label_delta, add_bbox_dummy=False):
        # compute comprehensive matching score based on matchig likelihood,
        # bbox confidence, and ious
        if add_bbox_dummy:
            bbox_iou_dummy = torch.ones(bbox_ious.size(0), 1,
                                        device=torch.cuda.current_device()) * self.bbox_dummy_iou
            bbox_ious = torch.cat((bbox_iou_dummy, bbox_ious), dim=1)
            label_dummy = torch.ones(bbox_ious.size(0), 1,
                                     device=torch.cuda.current_device())
            label_delta = torch.cat((label_dummy, label_delta), dim=1)
        if self.match_coeff is None:
            return match_ll
        else:
            # match coeff needs to be length of 3
            assert (len(self.match_coeff) == 3)
            return match_ll + self.match_coeff[0] * \
                   torch.log(bbox_scores) + self.match_coeff[1] * bbox_ious \
                   + self.match_coeff[2] * label_delta

    def forward(self, x, ref_x, x_n, ref_x_n):
        # x and ref_x are the grouped bbox features of current and reference frame
        # x_n are the numbers of proposals in the current images in the mini-batch,
        # ref_x_n are the numbers of ground truth bboxes in the reference images.
        # here we compute a correlation matrix of x and ref_x
        # we also add a all 0 column denote no matching
        assert len(x_n) == len(ref_x_n)
        # if self.with_avg_pool:
        #     x = self.avg_pool(x)
        #     ref_x = self.avg_pool(ref_x)
        x = x.view(x.size(0), -1)
        ref_x = ref_x.view(ref_x.size(0), -1)
        for idx, fc in enumerate(self.fcs):
            x = fc(x)
            ref_x = fc(ref_x)
            if idx < len(self.fcs) - 1:
                x = self.relu(x)
                ref_x = self.relu(ref_x)
        n = len(x_n)
        x_split = torch.split(x, x_n, dim=0)
        ref_x_split = torch.split(ref_x, ref_x_n, dim=0)
        prods = []
        for i in range(n):
            prod = torch.mm(x_split[i], torch.transpose(ref_x_split[i], 0, 1))
            prods.append(prod)
        match_score = []
        for prod in prods:
            m = prod.size(0)
            dummy = torch.zeros(m, 1, device=torch.cuda.current_device())

            prod_ext = torch.cat([dummy, prod], dim=1)
            match_score.append(prod_ext)
        return match_score

    def loss(self, match_score, ids, id_weights, reduce=True):
        losses = dict()
        n = len(match_score)
        x_n = [s.size(0) for s in match_score]
        # ids = torch.split(ids, x_n, dim=0)
        loss_match = 0.
        match_acc = 0.
        n_total = 0
        batch_size = len(ids)
        for score, cur_ids, cur_weights in zip(match_score, ids, id_weights):
            valid_idx = torch.nonzero(cur_weights).squeeze()
            if len(valid_idx.size()) == 0: continue
            n_valid = valid_idx.size(0)
            n_total += n_valid
            loss_match += weighted_cross_entropy(
                score, cur_ids, cur_weights, reduce=reduce)
            # match_acc += accuracy(torch.index_select(score, 0, valid_idx),
            #                       torch.index_select(cur_ids, 0, valid_idx)) * n_valid
        losses['loss_match'] = loss_match / n
        # if n_total > 0:
        #     losses['match_acc'] = match_acc / n_total
        return losses


class YolactTracking(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.total_cfg = cfg
        self.cfg = cfg.model.yolact_tracking
        self.yolact = Yolact(cfg)
        ckpt = torch.load(self.cfg.pretrained_yolact, 'cpu')
        ckpt = {k.lstrip("model."): v for k, v in ckpt['model'].items()}
        self.yolact.load_state_dict(ckpt)
        self.roi_align = RoIAlign((self.cfg.track_head.roi_feat_size,
                                   self.cfg.track_head.roi_feat_size), 69.0 / 550, 0)
        self.track_head = TrackHead(cfg)
        self.dbg = cfg.dbg is True

    def forward(self, dps):
        evaltime = EvalTime()
        ##############  ↓ Step : forward yolact for each frame  ↓  ##############
        evaltime('')
        assert self.yolact.training is False
        with torch.no_grad():
            preds0, feat0 = self.yolact({'image': dps['image0']}, return_features=True)
            preds1, feat1 = self.yolact({'image': dps['image1']}, return_features=True)
        preds0 = self.decode_yolact_preds(preds0, dps)
        preds1 = self.decode_yolact_preds(preds1, dps)
        evaltime('pred bbox')
        ##############  ↓ Step : roi align features  ↓  ##############
        feat0 = feat0[0]
        feat1 = feat1[0]
        roi_features0 = self.extract_roi_features(preds0, feat0)
        roi_features1 = self.extract_roi_features(preds1, feat1)
        evaltime('roi align')
        ##############  ↓ Step : forward track head  ↓  ##############
        ref_x = roi_features0
        ref_x_n = [len(a) for a in preds0]
        x = roi_features1
        x_n = [len(a) for a in preds1]
        match_score = self.track_head(x, ref_x, x_n, ref_x_n)
        evaltime('head forward')
        ##############  ↓ Step : make gt and compute loss  ↓  ##############
        ids, id_weights = self.prepare_targets(preds0, preds1, dps)
        evaltime('prepare targets')
        loss_dict = self.track_head.loss(match_score, ids, id_weights)
        evaltime('loss')
        return dps, loss_dict

    def decode_yolact_preds(self, preds, dps):
        img = dps['image0'][0]
        _, h, w = img.shape
        boxes = []
        for pred in preds:
            if 'box' in pred['detection']:
                bbox = pred['detection']['box']
                bbox[:, 0] *= w
                bbox[:, 1] *= h
                bbox[:, 2] *= w
                bbox[:, 3] *= h
            else:
                bbox = torch.empty([0, 4], dtype=torch.float, device='cuda')
            boxes.append(BoxList(bbox, image_size=[w, h]))
        return boxes

    def extract_roi_features(self, preds, feat):
        batchids = torch.cat([torch.full((len(boxlist), 1), i) for i, boxlist in enumerate(preds)]).cuda()
        roi_region = torch.cat([boxlist.bbox for boxlist in preds], dim=0)
        rois = torch.cat([batchids, roi_region], dim=1)
        roi_features = self.roi_align(feat, rois)
        return roi_features

    def prepare_targets(self, preds0, preds1, dps):
        evaltime = EvalTime()
        targets0 = dps['target0']
        targets1 = dps['target1']
        ids, id_weights = [], []
        assert len(preds0) == len(targets0)
        assert len(preds1) == len(targets1)
        for i in range(len(preds1)):
            if self.dbg: img1 = untsfm(dps['image1'][i], dps['width'][i].item(), dps['height'][i].item())
            if self.dbg: img0 = untsfm(dps['image0'][i], dps['width'][i].item(), dps['height'][i].item())
            pred1 = preds1[i]
            target1 = targets1[i]
            pred1 = pred1.resize(target1.size)
            if self.dbg: pred1.plot(img=img1, show=True)
            iou = boxlist_iou(pred1, target1)
            maxiou, idxs = iou.max(1)
            valid = maxiou > 0.7
            matched_targets = target1[idxs]
            if self.dbg: matched_targets.plot(img=img1, show=True)
            trackids1 = matched_targets.get_field('trackids').tolist()
            target0 = targets0[i]
            trackids0 = target0.get_field('trackids').tolist()
            selected_idx0 = []
            for j, tid1 in enumerate(trackids1):
                if tid1 in trackids0:
                    selected_idx0.append(trackids0.index(tid1))
                else:
                    selected_idx0.append(0)
                    valid[j] = False
            matched_target0 = target0[selected_idx0]
            if self.dbg: matched_target0.plot(img=img0, show=True)
            pred0 = preds0[i].resize(matched_target0.size)
            iou = boxlist_iou(matched_target0, pred0)
            maxiou, idxs = iou.max(1)
            valid = valid & (maxiou > 0.7)
            pred0 = pred0[idxs]
            if self.dbg: pred0.plot(img=img0, show=True)
            idxs = idxs + 1
            idxs[~valid] = 0
            ids.append(idxs)
            id_weights.append((idxs > 0).float())
        return ids, id_weights

    def train(self, mode=True):
        super(YolactTracking, self).train(mode)
        self.yolact.train(mode=False)


def untsfm(img, w, h):
    img_numpy = img.permute(1, 2, 0).cpu().numpy()
    img_numpy = img_numpy[:, :, (2, 1, 0)]  # To BRG
    means = [103.94, 116.78, 123.68]
    std = [57.38, 57.12, 58.40]
    # if cfg.backbone.transform.normalize:
    img_numpy = (img_numpy * np.array(std) + np.array(means)) / 255.0
    img_numpy = img_numpy[:, :, (2, 1, 0)]  # To RGB
    img_numpy = np.clip(img_numpy, 0, 1)
    img = cv2.resize(img_numpy, (w, h))
    return img


def weighted_cross_entropy(pred, label, weight, avg_factor=None,
                           reduce=True):
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)
    raw = F.cross_entropy(pred, label, reduction='none')
    if reduce:
        return torch.sum(raw * weight)[None] / avg_factor
    else:
        return raw * weight / avg_factor


def accuracy(pred, target, topk=1):
    if isinstance(topk, int):
        topk = (topk,)
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    _, pred_label = pred.topk(maxk, 1, True, True)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res
