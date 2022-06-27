from dl_ext.vision_ext.datasets.kitti.io import *
from tqdm import trange

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture


def main():
    CLS = 'Pedestrian'
    KITTI_ROOT = osp.expanduser('~/Datasets/kitti')
    height, center_disp = [], []
    for i in trange(7481):
        label2s = load_label_2(KITTI_ROOT, 'training', i, [CLS])
        label3s = load_label_3(KITTI_ROOT, 'training', i, [CLS])
        assert len(label2s) == len(label3s)
        for l2, l3 in zip(label2s, label3s):
            h = l2.y2 - l2.y1
            c2 = (l2.x1 + l2.x2) / 2
            c3 = (l3.x1 + l3.x2) / 2
            cd = c2 - c3
            height.append(h)
            center_disp.append(cd)
    plt.scatter(height, center_disp)
    plt.show()
    height = np.array(height)
    center_disp = np.array(center_disp)
    linear_regression = LinearRegression().fit(height.reshape(-1, 1), center_disp.reshape(-1, 1))
    print(linear_regression.coef_[0][0])
    print(linear_regression.intercept_[0])
    n = len(height)
    errs = center_disp - linear_regression.coef_[0][0] * height - linear_regression.intercept_[0]
    plt.hist(errs, 100)
    plt.show()
    # sigma2 = 1.0 / (n - 1) * (errs ** 2).sum()
    # sigma = sigma2 ** 0.5
    # print(sigma)
    gmm = GaussianMixture()
    r = gmm.fit(errs[:, np.newaxis])
    print("mean : %f, var : %f" % (r.means_[0, 0], r.covariances_[0, 0]))
    print('std: ', r.covariances_[0, 0] ** 0.5)


##############  ↓  Car  ↓  ##############
# 0.18742550637410094
# 4.41360665450609
# mean : -0.000000, var : 567.156004
# std:  [23.81503735]

##############  ↓ Step : Pedestrain  ↓  ##############
# 0.27261221916974204
# 1.707562061026895
# mean : -0.000000, var : 80.241872
# std:  [8.95778275]


if __name__ == '__main__':
    main()
