import numpy as np
from skimage import measure
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, average_precision_score

def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


def is_one_class(gt: np.ndarray):
    gt_ravel = gt.ravel()
    return gt_ravel.sum() == 0 or gt_ravel.sum() == gt_ravel.shape[0]


def calculate_px_metrics(gt_px, pr_px):
    if is_one_class(gt_px):  # In case there are only normal pixels or no pixel-level labels
        return 0, 0, 0, 0

    auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())
    precisions, recalls, _ = precision_recall_curve(gt_px.ravel(), pr_px.ravel())
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    f1_px = np.max(f1_scores[np.isfinite(f1_scores)])
    ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())

    gt_px = gt_px.squeeze()
    pr_px = pr_px.squeeze()
    aupro_px = cal_pro_score(gt_px, pr_px)

    return auroc_px * 100, f1_px * 100, ap_px * 100, aupro_px * 100


def calculate_im_metrics(gt_im, pr_im):
    if is_one_class(gt_im):  # In case there are only normal samples or no image-level labels
        return 0, 0, 0

    auroc_im = roc_auc_score(gt_im.ravel(), pr_im.ravel())
    precisions, recalls, _ = precision_recall_curve(gt_im.ravel(), pr_im.ravel())
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    f1_im = np.max(f1_scores[np.isfinite(f1_scores)])
    ap_im = average_precision_score(gt_im, pr_im)

    return ap_im * 100, auroc_im * 100, f1_im * 100


def calculate_average_metric(metrics: dict):
    average = {}
    for obj, metric in metrics.items():
        for k, v in metric.items():
            if k not in average:
                average[k] = []
            average[k].append(v)

    for k, v in average.items():
        average[k] = np.mean(v)

    return average


def calculate_metric(results, obj):
    gt_px = []
    pr_px = []

    gt_im = []
    pr_im = []

    for idx in range(len(results['cls_names'])):
        if results['cls_names'][idx] == obj:
            gt_px.append(results['imgs_masks'][idx])
            pr_px.append(results['anomaly_maps'][idx])

            gt_im.append(results['imgs_gts'][idx])
            pr_im.append(results['anomaly_scores'][idx])

    gt_px = np.array(gt_px)
    pr_px = np.array(pr_px)

    gt_im = np.array(gt_im)
    pr_im = np.array(pr_im)

    auroc_px, f1_px, ap_px, aupro_px = calculate_px_metrics(gt_px, pr_px)
    ap_im, auroc_im, f1_im = calculate_im_metrics(gt_im, pr_im)

    metric = {
        'auroc_px': auroc_px,
        'auroc_im': auroc_im,
        'f1_px': f1_px,
        'f1_im': f1_im,
        'ap_px': ap_px,
        'ap_im': ap_im,
        'aupro_px': aupro_px
    }

    return metric
