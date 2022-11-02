import torch

from disprcnn.utils.pytorch_ssim import SSIM
from disprcnn.utils.timer import EvalTime

lp, rp, img2, img3 = torch.load('tmp/lprp_input.pth')
evaltime = EvalTime()
ssim = SSIM().cuda()
for i in range(10000):
    evaltime("")
    W, H = lp.size
    lboxes = lp.bbox.round().long()
    rboxes = rp.bbox.round().long()
    ssims = torch.full((len(lboxes), len(rboxes)), -10.0)
    ssim_coef = 0.18742550637410094
    ssim_intercept = 4.41360665450609
    ssim_std = 23.81503735
    M, N = lboxes.shape[0], rboxes.shape[0]
    lboxes_exp = lboxes.unsqueeze(1).repeat(1, N, 1)
    rboxes_exp = rboxes.unsqueeze(0).repeat(M, 1, 1)
    hmeans = (lboxes_exp[:, :, 3] - lboxes_exp[:, :, 1] + rboxes_exp[:, :, 3] - rboxes_exp[:, :, 1]) / 2
    center_disp_expectations = hmeans * ssim_coef + ssim_intercept
    cds = (lboxes_exp[:, :, 0] + lboxes_exp[:, :, 2] - rboxes_exp[:, :, 0] - rboxes_exp[:, :, 2]) / 2
    valid = (cds - center_disp_expectations).abs() < 3 * ssim_std
    nzs = valid.nonzero()
    lrois, rrois = [], []
    sls, srs = lboxes[nzs[:, 0]], rboxes[nzs[:, 1]]
    w = torch.max(sls[:, 2] - sls[:, 0], srs[:, 2] - srs[:, 0])
    h = torch.max(sls[:, 3] - sls[:, 1], srs[:, 3] - srs[:, 1])
    ws = torch.min(torch.min(w, W - sls[:, 0]), W - srs[:, 0])
    hs = torch.min(torch.min(h, H - sls[:, 1]), H - srs[:, 1])
    evaltime("match_lr: prepare")
    for nz, w, h in zip(nzs, ws, hs):
        i, j = nz
        x1, y1, x2, y2 = lboxes[i]
        x1p, y1p, x2p, y2p = rboxes[j]
        lroi = img2[y1:y1 + h, x1:x1 + w, :].permute(2, 0, 1)[None] / 255.0
        rroi = img3[y1p:y1p + h, x1p:x1p + w, :].permute(2, 0, 1)[None] / 255.0
        lrois.append(lroi)
        rrois.append(rroi)
    evaltime("match_lr: crop loop")
    for nz, lroi, rroi in zip(nzs, lrois, rrois):
        i, j = nz
        s = ssim(lroi, rroi)
        ssims[i, j] = s
    # print(ssims)
    # break
    evaltime("match_lr: ssim loop")
    continue
    if len(lboxes) <= len(rboxes):
        num = ssims.shape[0]
    else:
        num = ssims.shape[1]
    lidx, ridx = [], []
    for _ in range(num):
        tmp = torch.argmax(ssims).item()
        row, col = tmp // ssims.shape[1], tmp % ssims.shape[1]
        if ssims[row, col] > 0:
            lidx.append(row)
            ridx.append(col)
        ssims[row] = ssims[row].clamp(max=0)
        ssims[:, col] = ssims[:, col].clamp(max=0)
    lp = lp[lidx]
    rp = rp[ridx]
    evaltime("match_lr: ready to return")
