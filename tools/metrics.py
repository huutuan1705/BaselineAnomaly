import numpy as np
from skimage import measure
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, average_precision_score

def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # Same thresholds and same PRO / FPR definition as the reference, but every pixel is visited once:
    # a value v is mapped to the number of thresholds strictly below it, so "v > ths[k]" <=> "below > k".
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    ths = np.arange(min_th, max_th, delta)
    n_th = len(ths)

    normal_hist = np.zeros(n_th + 1, dtype=np.int64)  # normal pixels, counted by `below`
    region_pros = []  # one entry per GT region: its PRO at every threshold
    for amap, mask in zip(amaps, masks):
        below = np.searchsorted(ths, amap.ravel(), side='left').reshape(amap.shape)
        normal_hist += np.bincount(below[mask == 0], minlength=n_th + 1)
        if mask.any():
            for region in measure.regionprops(measure.label(mask)):
                hist = np.bincount(below[region.coords[:, 0], region.coords[:, 1]], minlength=n_th + 1)
                tp_pixels = region.area - np.cumsum(hist)[:n_th]  # region pixels with value > ths[k]
                region_pros.append(tp_pixels / region.area)

    n_normal = normal_hist.sum()
    fprs = (n_normal - np.cumsum(normal_hist)[:n_th]) / n_normal  # normal pixels with value > ths[k]
    if region_pros:
        region_pros = np.ascontiguousarray(np.array(region_pros).T)  # (n_th, n_regions)
        pros = np.array([pro.mean() for pro in region_pros])
    else:
        pros = np.full(n_th, np.nan)
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

def cal_px_auroc_f1_ap(gt, pr):
    is_pos = gt > 0
    pos = np.sort(pr[is_pos])
    neg = np.sort(pr[~is_pos])
    n_pos, n_neg = pos.size, neg.size

    # AUROC = P(pos > neg) + 0.5 * P(pos == neg), i.e. the trapezoidal area under the ROC curve
    n_below = np.searchsorted(neg, pos, side='left').sum(dtype=np.int64)
    n_below_or_equal = np.searchsorted(neg, pos, side='right').sum(dtype=np.int64)
    auroc = (n_below + n_below_or_equal) / (2.0 * n_pos * n_neg)

    # Recall only changes at thresholds equal to a positive score, so these are the only points
    # that contribute to AP and the only candidates for the best F1.
    ths, counts = np.unique(pos, return_counts=True)
    ths, counts = ths[::-1], counts[::-1]  # decreasing threshold
    tps = np.cumsum(counts)  # positives with score >= threshold
    fps = n_neg - np.searchsorted(neg, ths, side='left')  # negatives with score >= threshold
    precisions = tps / (tps + fps)
    recalls = tps / n_pos
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    ap = np.sum(np.diff(recalls, prepend=0.0) * precisions)

    return auroc, np.max(f1_scores), ap

def calculate_px_metrics(gt_px, pr_px):
    if is_one_class(gt_px):  # In case there are only normal pixels or no pixel-level labels
        return 0, 0, 0, 0

    auroc_px, f1_px, ap_px = cal_px_auroc_f1_ap(gt_px.ravel(), pr_px.ravel())

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
