import numpy as np
import torch
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve


def fpr_at_tpr_x(y_true, score, x=0.95):
    fpr, tpr, _ = roc_curve(y_true, score)
    return fpr[(np.abs(tpr - x)).argmin()]


def evaluate_score(y, score):
    if score.dtype == torch.bfloat16:
        score = score.to(torch.float32)

    precision, recall, _ = precision_recall_curve(y, score)
    return {
        'FPR': fpr_at_tpr_x(y, score).item(),
        'AUROC': roc_auc_score(y, score).item(),
        'AUPR': auc(recall, precision).item(),
    }


def evaluate_ood(
    id_uncerts,
    ood_uncerts,
):
    all_uncerts = torch.cat([id_uncerts, ood_uncerts], dim=0)
    all_ood_classes = torch.cat([torch.zeros(len(id_uncerts)), torch.ones(len(ood_uncerts))], dim=0)
    # compute roc_auc for the ood task
    retval = evaluate_score(all_ood_classes, all_uncerts.float())
    return retval

