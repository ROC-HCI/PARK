import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, roc_auc_score,
    f1_score, recall_score, precision_score, confusion_matrix,
    brier_score_loss
)

def safe_divide(a, b, eps=1e-8):
    return a / b if b else 0.0

def expected_calibration_error_v1(y, y_pred_scores, num_buckets=10):
    y_pred_scores = np.asarray(y_pred_scores).flatten()
    
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, num_buckets + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i
    confidences = np.maximum(y_pred_scores, 1.0-y_pred_scores)

    # get predictions from confidences (positional in this case)
    predicted_label = (y_pred_scores>=0.5)
    
    # get a boolean list of correct/false predictions
    accuracies = (predicted_label==y)

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece.item()

def expected_calibration_error(confidences, labels, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins+1)
    ece = 0.0
    preds = (confidences >= 0.5).astype(int)
    correct = (preds == labels)
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        if np.any(mask):
            acc = correct[mask].mean()
            conf = confidences[mask].mean()
            ece += np.abs(acc - conf) * mask.mean()
    return ece

def compute_metrics(
    y_pred_scores,
    y_true,
    threshold=0.5,
    n_bootstraps=1000,
    n_bins=10,  
    random_state=None
):

    labels = np.asarray(y_true).reshape(-1)
    scores = np.asarray(y_pred_scores).reshape(-1)
    preds  = scores >= threshold
    n = len(labels)

    def _single_metrics(lab, scr, pr):
        out = {}
        out['accuracy']          = accuracy_score(lab, pr)
        out['average_precision'] = average_precision_score(lab, scr)
        out['auroc']             = roc_auc_score(lab, scr)
        out['f1_score']          = f1_score(lab, pr)
        tn, fp, fn, tp = confusion_matrix(lab, pr).ravel()
        out['tn'], out['fp'], out['fn'], out['tp'] = int(tn), int(fp), int(fn), int(tp)
        out['weighted_accuracy'] = (safe_divide(tp, tp + fp) + safe_divide(tn, tn + fn)) / 2.0
        out['recall']       = recall_score(lab, pr)        # sensitivity
        out['FPR']          = safe_divide(fp, fp + tn)
        out['precision']    = precision_score(lab, pr)     # PPV
        out['NPV']          = safe_divide(tn, tn + fn)
        out['specificity']  = safe_divide(tn, tn + fp)
        return out

    # 1) Point estimates
    metrics = _single_metrics(labels, scores, preds)

    # 2) Bootstrap distributions
    rng = np.random.RandomState(random_state)
    bootables = {k: [] for k in metrics if k not in ('tn','fp','fn','tp')}

    for _ in range(n_bootstraps):
        idx   = rng.randint(0, n, n)
        lab_b = labels[idx]
        scr_b = scores[idx]
        pr_b  = scr_b >= threshold
        mb    = _single_metrics(lab_b, scr_b, pr_b)
        for k in bootables:
            bootables[k].append(mb[k])

# 3) Build CIs using normal approximation (mu ± 1.96 * SE)
    for k, vals in bootables.items():
        vals = np.array(vals)
        mu = vals.mean()
        se = np.sqrt(((vals - mu) ** 2).sum() / (len(vals) - 1))
        lo = mu - 1.96 * se
        hi = mu + 1.96 * se
        metrics[f"{k}_ci"] = (float(round(lo, 4)), float(round(hi, 4)))

    # 4) Cast any np scalars to native
    for k, v in list(metrics.items()):
        if isinstance(v, np.generic):
            metrics[k] = v.item()

    # 5) Repackage confusion matrix
    metrics['confusion_matrix'] = {
        'tn': metrics.pop('tn'),
        'fp': metrics.pop('fp'),
        'fn': metrics.pop('fn'),
        'tp': metrics.pop('tp')
    }

    # 6) Formatted percent strings
    def _format_pct(name, mean_dp=2, half_dp=1):
        mean = metrics[name] * 100
        lo, hi = metrics[f"{name}_ci"]
        half = (hi - lo) / 2 * 100
        return f"{mean:.{mean_dp}f} ± {half:.{half_dp}f}%"

    metrics['formatted_accuracy']    = _format_pct('accuracy')
    metrics['formatted_sensitivity'] = _format_pct('recall')
    metrics['formatted_specificity'] = _format_pct('specificity')
    metrics['formatted_ppv']         = _format_pct('precision')
    metrics['formatted_npv']         = _format_pct('NPV')
    metrics['formatted_f1score']     = _format_pct('f1_score')

    # 7) Plain error strings (half-width only, rounded to 2 decimals)
    err_map = {
        'accuracy':    'accuracy_err',
        'recall':      'sensitivity_err',
        'specificity': 'specificity_err',
        'precision':   'ppv_err',
        'NPV':         'npv_err',
        'f1_score':    'f1score_err',
    }
    for metric_name, err_key in err_map.items():
        lo, hi = metrics[f"{metric_name}_ci"]
        half = (hi - lo) / 2        # raw fraction
        metrics[err_key] = half     # e.g. 0.0123
    
    metrics['Brier Score'] = brier_score_loss(labels, scores)
    metrics['ECE']         = expected_calibration_error_v1(labels, scores, num_buckets=n_bins)

        
    return metrics
