import torch

from disprcnn.utils.pytorch_ssim import SSIM
from disprcnn.utils.timer import EvalTime

lp, rp, img2, img3 = torch.load('tmp/lprp_input.pth')
evaltime = EvalTime()
ssim = SSIM().cuda()
for i in range(10000):
    evaltime("")
    W, H = lp.size
    lboxes = lp.bbox.round().long().tolist()
    rboxes = rp.bbox.round().long().tolist()
    ssims = torch.zeros((len(lboxes), len(rboxes)))
    # ssim_coefs: [  ]
    # ssim_intercepts: [ 4.41360665450609 ]
    # ssim_stds: [ 23.81503735 ]
    ssim_coef = 0.18742550637410094
    ssim_intercept = 4.41360665450609
    ssim_std = 23.81503735
    evaltime("match_lr: prepare")
    for i in range(len(lboxes)):
        x1, y1, x2, y2 = lboxes[i]
        for j in range(len(rboxes)):
            x1p, y1p, x2p, y2p = rboxes[j]
            # adaptive thresh
            hmean = (y2 - y1 + y2p - y1p) / 2
            center_disp_expectation = hmean * ssim_coef + ssim_intercept
            cd = (x1 + x2 - x1p - x2p) / 2
            if abs(cd - center_disp_expectation) < 3 * ssim_std:
                w = max(x2 - x1, x2p - x1p)
                h = max(y2 - y1, y2p - y1p)
                w = min(min(w, W - x1, ), W - x1p)
                h = min(min(h, H - y1, ), H - y1p)
                lroi = img2[y1:y1 + h, x1:x1 + w, :].permute(2, 0, 1)[None] / 255.0
                rroi = img3[y1p:y1p + h, x1p:x1p + w, :].permute(2, 0, 1)[None] / 255.0
                s = ssim(lroi, rroi)
            else:
                s = -10
            ssims[i, j] = s
    print(ssims)
    break
    evaltime("match_lr: for loop")
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
