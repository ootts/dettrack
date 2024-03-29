import io

import imageio
import tempfile

import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

from disprcnn.structures.segmentation_mask import SegmentationMask

from disprcnn.utils.averagemeter import AverageMeter

from disprcnn.utils.timer import EvalTime
from torchvision.ops import RoIAlign
from matplotlib import colors as mcolors
from disprcnn.structures.bounding_box import BoxList

from disprcnn.modeling.models.yolact.submodules import *
from disprcnn.structures.boxlist_ops import boxlist_iou, cat_boxlist
from disprcnn.utils.vis3d_ext import Vis3D
from .layers.output_utils import postprocess
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
        assert len(x_n) == len(ref_x_n)
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
        loss_match = 0.0
        match_acc = 0.0
        n_total = 0
        batch_size = len(ids)
        for score, cur_ids, cur_weights in zip(match_score, ids, id_weights):
            valid_idx = torch.nonzero(cur_weights).squeeze(1)
            if valid_idx.numel() == 0: continue
            n_valid = valid_idx.size(0)
            n_total += n_valid
            loss_match += weighted_cross_entropy(
                score, cur_ids, cur_weights, reduce=reduce)
            match_acc += accuracy(torch.index_select(score, 0, valid_idx),
                                  torch.index_select(cur_ids, 0, valid_idx)) * n_valid
        if isinstance(loss_match, float):
            loss_match = torch.tensor([0.0], requires_grad=True)
        if isinstance(match_acc, float):
            match_acc = torch.tensor([0.0], requires_grad=True)
        losses['loss_match'] = loss_match / n
        if n_total > 0:
            match_acc = match_acc / n_total
        return losses['loss_match'], match_acc


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
        self.seq_id = -1
        self.memory = None
        self.boxmemory = [0]
        self.evaltime = EvalTime(disable=not cfg.evaltime, do_print=False)
        self.meter_2d = AverageMeter()
        self.meter_track = AverageMeter()

    def forward_train(self, dps):
        ##############  ↓ Step : forward yolact for each frame  ↓  ##############
        self.evaltime('')
        assert self.yolact.training is False
        with torch.no_grad():
            preds0, feat0 = self.yolact({'image': dps['image0']}, return_features=True)
            preds1, feat1 = self.yolact({'image': dps['image1']}, return_features=True)
        preds0 = self.decode_yolact_preds(preds0, dps['image0'].shape[-2], dps['image0'].shape[-1])
        preds1 = self.decode_yolact_preds(preds1, dps['image0'].shape[-2], dps['image0'].shape[-1])
        self.evaltime('pred bbox')
        ##############  ↓ Step : roi align features  ↓  ##############
        feat0 = feat0[0]
        feat1 = feat1[0]
        roi_features0 = self.extract_roi_features(preds0, feat0)
        roi_features1 = self.extract_roi_features(preds1, feat1)
        self.evaltime('roi align')
        ##############  ↓ Step : forward track head  ↓  ##############
        ref_x = roi_features0
        if len(ref_x) == 0:
            output = {'metrics': {'acc': torch.tensor([0.0], requires_grad=True)}}
            loss_dict = {'loss_match': torch.tensor([0.0], requires_grad=True)}
            return output, loss_dict
        ref_x_n = [len(a) for a in preds0]
        x = roi_features1
        x_n = [len(a) for a in preds1]
        match_score = self.track_head(x, ref_x, x_n, ref_x_n)
        self.evaltime('head forward')
        ##############  ↓ Step : make gt and compute loss  ↓  ##############
        ids, id_weights = self.prepare_targets(preds0, preds1, dps)
        self.evaltime('prepare targets')
        loss_match, acc = self.track_head.loss(match_score, ids, id_weights)

        output = {'metrics': {'acc': acc}}
        loss_dict = {'loss_match': loss_match}
        self.evaltime('loss')
        return output, loss_dict

    def calcIOUscore(self, preds):
        if not isinstance(self.boxmemory, BoxList):
            return torch.empty([len(preds[0]), len(self.memory)], dtype=torch.float, device='cuda')
        ious = torch.zeros([len(preds[0]), 1], dtype=torch.float, device="cuda")
        ious = torch.cat([ious, boxlist_iou(preds[0], self.boxmemory)], dim=1)
        return ious

    def forward_test(self, dps, track=True, add_mask=False, mask_mode='poly'):
        vis3d = Vis3D(
            xyz_pattern=('x', 'y', 'z'),
            out_folder="dbg",
            sequence="yolact_tracking_forward_test",
            # auto_increase=,
            enable=self.dbg,
        )
        self.evaltime('2d begin')
        preds, feat = self.yolact({'image': dps['image']}, return_features=True)
        # if preds[0]['detection'] is not None:
        #     confidence = preds[0]["detection"]["score"]
        preds = self.decode_yolact_preds(preds, dps['image'].shape[-2], dps['image'].shape[-1], add_mask=add_mask,
                                         mask_mode=mask_mode)
        confidence = preds[0].get_field('scores')
        if self.total_cfg.evaltime:
            self.meter_2d.update(self.evaltime('2d end'))
            # print('2d', self.meter_2d.avg)
        width, height = dps['width'][0].item(), dps['height'][0].item()

        if track:
            seqid = dps['seq']
            assert seqid.shape[0] == 1  # batch size=1
            seqid = seqid[0].item()
            if seqid != self.seq_id:
                # reset
                self.seq_id = seqid
                self.memory = torch.empty([0, 256, 7, 7], dtype=torch.float, device='cuda')  # todo: put in cfg
                self.boxmemory = None
            feat = feat[0]
            roi_features = self.extract_roi_features(preds, feat)
            ref_x = self.memory
            ref_x_n = [len(self.memory)]
            x = roi_features
            x_n = [len(a) for a in preds]
            if len(self.memory) > 0 and x.numel() > 0:
                match_score = self.track_head(x, ref_x, x_n, ref_x_n)[0]
            else:
                match_score = torch.empty([len(preds[0]), len(self.memory)], dtype=torch.float, device='cuda')
            ##############  ↓ Step : match  ↓  ##############
            if match_score.numel() > 0:
                match_score = F.softmax(match_score, dim=1)
                iou_score = self.calcIOUscore(preds)
                conf_score = torch.t(confidence.repeat(len(self.boxmemory) + 1, 1))
                match_score = torch.log(match_score) + self.cfg.alpha * torch.log(
                    conf_score) + self.cfg.beta * iou_score
                # TODO
                max_score, idxs = match_score.max(1)
                # todo fix bug for empty bin!!!
                # dual = match_score.max(0).indices[match_score.max(1).indices] == torch.arange(len(preds[0])).cuda()
                # matched = max_score > 0.5
                matched = (max_score > self.cfg.thresh) & (idxs != 0)
                unmatched = ~matched
                matchidx = idxs[matched]
                duplicateid = []
                for id in matchidx:
                    if id == 0:
                        continue
                    if sum(matchidx == id) > 1:
                        if id not in duplicateid:
                            duplicateid.append(id)
                for id in duplicateid:
                    conflict_box = (idxs == id) & matched
                    idscores = match_score[..., id]
                    idscores[~conflict_box] = -torch.inf
                    maxval, maxid = idscores.max(0)
                    conflict_box[maxid] = False
                    matched[conflict_box] = False

                matchidx = idxs[matched]

                # matched = matched & dual
            else:
                matched = torch.full([len(preds[0])], False).cuda()
                unmatched = ~matched
            cur_trackids = torch.full([len(preds[0])], -1).long().cuda()
            if matched.sum() > 0:
                idxs = idxs - 1
                cur_trackids[matched] = idxs[matched]
                cur_feat = roi_features[matched]
                self.memory[idxs[matched]] = cur_feat
                # todo:simplify
                self.boxmemory.bbox[idxs[matched]] = preds[0].bbox[matched]

            if unmatched.sum() > 0:
                new_tids = torch.arange(self.memory.shape[0], self.memory.shape[0] + unmatched.sum()).long().cuda()
                cur_trackids[unmatched] = new_tids
                self.memory = torch.cat([self.memory, roi_features[unmatched]], dim=0)
                # res_box = preds[0][unmatched]
                if not isinstance(self.boxmemory, BoxList):
                    self.boxmemory = preds[0][unmatched]
                else:
                    self.boxmemory = cat_boxlist([self.boxmemory, preds[0][unmatched]], ignore_fields=True,
                                                 ignore_maps=True)
                    # if isinstance(self.boxmemory, BoxList):
                #     self.boxmemory = cat_boxlist([self.boxmemory, res_box])
                # else:
                #     self.boxmemory = res_box
            keep = matched | unmatched

            pred = preds[0].resize([width, height])
            if pred.has_map('masks'):
                pred.add_map('masks', pred.get_map('masks').convert('mask').resize([width, height]))
            pred = pred[keep]
            cur_trackids = cur_trackids[keep]

            pred.add_field('trackids', cur_trackids)
        else:
            pred = preds[0].resize([width, height])
        if self.total_cfg.evaltime:
            self.meter_track.update(self.evaltime('track'))
            # print('track', self.meter_track.avg)
        if self.dbg:
            plt.title(f'global_step: {dps["global_step"]}')
            img = untsfm(dps['image'][0], width, height)
            plt.imshow(img)
            colors = list(mcolors.BASE_COLORS.keys())
            for i, box in enumerate(pred.convert('xywh').bbox.tolist()):
                x, y, w, h = box
                score = pred.get_field('scores').tolist()[i]
                if track:
                    trackid = pred.get_field("trackids")[i]
                    c = colors[trackid % len(colors)]
                    plt.text(x, y, f'{trackid}_{score:.2f}', color=c, fontsize='x-large')
                    plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=False, color=c, linewidth=2))
                else:
                    plt.gca().add_patch(
                        plt.Rectangle((x, y), w, h, fill=False, color=colors[i % len(colors)], linewidth=2))
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            vis3d.add_image(Image.open(img_buf))
            img_buf.close()
            plt.close("all")
        loss_dict = {}
        return pred, loss_dict

    def forward(self, dps, track=True, add_mask=False, mask_mode='poly'):
        if self.training:
            return self.forward_train(dps)
        else:
            return self.forward_test(dps, track=track, add_mask=add_mask, mask_mode=mask_mode)

    def decode_yolact_preds(self, preds, h, w, add_mask=False, retvalid=True, mask_mode='poly'):
        if self.total_cfg.model.meta_architecture == 'YolactTracking':
            boxes = []
            for pred in preds:
                if pred['detection'] is not None:
                    bbox = pred['detection']['box'].clone()
                    bbox[:, 0] *= w
                    bbox[:, 1] *= h
                    bbox[:, 2] *= w
                    bbox[:, 3] *= h
                    labels = pred['detection']['class'] + 1
                    scores = pred['detection']['score']
                else:
                    bbox = torch.empty([0, 4], dtype=torch.float, device='cuda')
                    labels = torch.empty([0, ], dtype=torch.long, device='cuda')
                    scores = torch.empty([0, ], dtype=torch.float, device='cuda')
                boxlist = BoxList(bbox, image_size=[w, h])
                boxlist.add_field("labels", labels)
                boxlist.add_field("scores", scores)
                boxes.append(boxlist)
            return boxes
        elif self.total_cfg.model.meta_architecture == 'DRCNN':
            if preds[0]['detection'] is not None:
                preds = postprocess(preds, w, h, to_long=False)
                labels, scores, box2d, masks = preds
            else:
                box2d = torch.empty([0, 4], dtype=torch.float, device='cuda')
                labels = torch.empty([0], dtype=torch.long, device='cuda')
                scores = torch.empty([0], dtype=torch.float, device='cuda')
                masks = torch.empty([0, h, w], dtype=torch.long, device='cuda')
            boxlist = BoxList(box2d, (w, h))
            boxlist.add_field("labels", labels + 1)
            boxlist.add_field("scores", scores)
            if add_mask:
                keep = masks.sum(1).sum(1) > 20
                boxlist = boxlist[keep]
                masks = masks[keep]
                if retvalid:
                    vec, masks = SegmentationMask(masks, (w, h), mode='mask').convert(mask_mode, retvalid=retvalid)
                    if vec is not None and not vec.all():
                        boxlist = boxlist[vec]
                else:
                    masks = SegmentationMask(masks, (w, h), mode='mask').convert(mask_mode, retvalid=retvalid)
                boxlist.add_map("masks", masks)
            return [boxlist]

    def extract_roi_features(self, preds, feat):
        batchids = torch.cat([torch.full((len(boxlist), 1), i) for i, boxlist in enumerate(preds)]).cuda()
        roi_region = torch.cat([boxlist.bbox for boxlist in preds], dim=0)
        rois = torch.cat([batchids, roi_region], dim=1)
        roi_features = self.roi_align(feat, rois)
        return roi_features

    def prepare_targets(self, preds0, preds1, dps):
        evaltime = EvalTime(disable=True)
        targets0 = dps['target0']
        targets1 = dps['target1']
        ids, id_weights = [], []
        assert len(preds0) == len(targets0)
        assert len(preds1) == len(targets1)
        for i in range(len(preds1)):
            evaltime('')
            if self.dbg: img1 = untsfm(dps['image1'][i], dps['width'][i].item(), dps['height'][i].item())
            if self.dbg: img0 = untsfm(dps['image0'][i], dps['width'][i].item(), dps['height'][i].item())
            pred1 = preds1[i]
            target1 = targets1[i]
            pred1 = pred1.resize(target1.size)
            pred0 = preds0[i].resize(target1.size)
            if len(pred0) == 0 or len(pred1) == 0:
                ids.append(torch.empty((0), dtype=torch.long, device='cuda'))
                id_weights.append(torch.empty((0), dtype=torch.float, device='cuda'))
                continue
            if self.dbg: pred1.plot(img=img1, show=True)
            iou = boxlist_iou(pred1, target1)
            maxiou, idxs = iou.max(1)
            valid = maxiou > 0.7
            matched_targets = target1[idxs]
            evaltime('pred1 and tgt1')
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
            evaltime('tgt1 and tgt0')
            if self.dbg: matched_target0.plot(img=img0, show=True)
            iou = boxlist_iou(matched_target0, pred0)
            maxiou, idxs = iou.max(1)
            valid = valid & (maxiou > 0.7)
            pred0 = pred0[idxs]
            if self.dbg: pred0.plot(img=img0, show=True)
            idxs = idxs + 1
            idxs[~valid] = 0
            evaltime('tgt0 and pred0')
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
