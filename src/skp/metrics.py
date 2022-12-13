import numpy as np
import torch
import pytorch_lightning as pl

import torchmetrics as tm
import torchmetrics.functional as FM
from sklearn.metrics import cohen_kappa_score, roc_auc_score, average_precision_score


def _roc_auc_score(t, p):
    try:
        return torch.tensor(roc_auc_score(t, p) if len(np.unique(t)) > 1 else 0.5)
    except ValueError:
        return torch.tensor(0.5)


def _average_precision_score(t, p):
    return torch.tensor(average_precision_score(t, p) if len(np.unique(t)) > 1 else 0)


def _dsc(p, t, **kwargs):
    p = (p>=0.5)
    t_sum = t.sum(-1)
    p_sum = p.sum(-1)
    return (2*(p*t).sum(-1)/((p+t).sum(-1)))


class _BaseMetric(tm.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("p", default=[], dist_reduce_fx=None)
        self.add_state("t", default=[], dist_reduce_fx=None)

    def update(self, p, t):
        self.p.append(p)
        self.t.append(t)

    def compute(self):
        raise NotImplementedError


class AUROC(_BaseMetric):
    """For simple binary classification
    """
    def compute(self):
        p = torch.cat(self.p, dim=0).cpu().numpy() #(N,C)
        t = torch.cat(self.t, dim=0).cpu().numpy() #(N,C)
        return {'auroc': _roc_auc_score(t, p)}


class AUROCmc(_BaseMetric):
    """For multilabel binary classification"""

    def compute(self):
        p = torch.cat(self.p, dim=0).cpu().numpy()  # (N,C)
        t = torch.cat(self.t, dim=0).cpu().numpy()  # (N,C)
        auc_dict = {}
        if p.ndim == 1:
            return {"auc_mean": _roc_auc_score(t, p)}
        for c in range(p.shape[1]):
            tmp_gt = t == c if t.ndim == 1 else t[:, c]
            auc_dict[f"auc{c}"] = _roc_auc_score(tmp_gt, p[:, c])
        auc_dict["auc_mean"] = np.mean([v for v in auc_dict.values()])
        return auc_dict

class AVP(_BaseMetric):
    """For simple binary classification
    """
    def compute(self):
        p = torch.cat(self.p, dim=0).cpu().numpy() #(N,C)
        t = torch.cat(self.t, dim=0).cpu().numpy() #(N,C)
        avp_dict = {}
        for c in range(p.shape[1]):
            tmp_gt = t == c if t.ndim == 1 else t[:,c]
            avp_dict[f'avp{c}'] = _average_precision_score(tmp_gt, p[:,c])
        avp_dict['avp_mean'] = np.mean([v for v in avp_dict.values()])
        return avp_dict


# class DSC(pl.metrics.Metric):

#     def __init__(self, dist_sync_on_step=False):
#         super().__init__(dist_sync_on_step=dist_sync_on_step)

#         self.add_state('dsc', default=[], dist_reduce_fx=None)
#         self.add_state('p_auc', default=[], dist_reduce_fx=None)
#         self.add_state('t_auc', default=[], dist_reduce_fx=None)

#     def update(self, p, t):
#         # p.shape = (N, C, H, W)
#         # t.shape = (N, C, H, W)
#         N,C = p.shape[:2]
#         self.C = C
#         p = torch.sigmoid(p)
#         p = p.reshape(N, C, -1)
#         t = t.reshape(N, C, -1)
#         dsc_list = []
#         for i in range(C):
#             dsc_list.append(_dsc(p[:,i], t[:,i]))
#         self.dsc.append(torch.stack(dsc_list, dim=1))
#         self.p_auc.append(p[:,0].max(1)[0])
#         self.t_auc.append(t[:,0].max(1)[0])

#     def compute(self):
#         p = torch.cat(self.p_auc, dim=0).cpu().numpy()
#         t = torch.cat(self.t_auc, dim=0).cpu().numpy()
#         dsc_scores = torch.cat(self.dsc, dim=0)
#         metrics_dict = {}
#         for i in range(self.C):
#             metrics_dict[f'dsc{i:02d}'] = dsc_scores[:,i].mean()
#         metrics_dict[f'auc_seg{i:02d}'] = _roc_auc_score(t, p)
#         metrics_dict['dsc'] = dsc_scores.mean()
#         return metrics_dict


class Accuracy(_BaseMetric):

    def compute(self): 
        p = torch.cat(self.p, dim=0)
        t = torch.cat(self.t, dim=0)
        return dict(accuracy=(p.argmax(1) == t).float().mean())


class Kappa(_BaseMetric):

    def compute(self): 
        p = torch.cat(self.p, dim=0).cpu().numpy()
        t = torch.cat(self.t, dim=0).cpu().numpy()
        return dict(kappa=cohen_kappa_score(t, np.argmax(p, axis=1)))