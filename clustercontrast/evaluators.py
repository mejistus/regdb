from __future__ import print_function, absolute_import
import time
import collections
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
import random
import copy

from .evaluation_metrics import cmc, mean_ap
from .utils.meters import AverageMeter
from .utils.rerank import re_ranking
from .utils import to_torch

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip


def _enable_dropout(model):
    """Enable dropout layers during evaluation for MC Dropout."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

def extract_cnn_feature(model, inputs,mode):
    inputs = to_torch(inputs).cuda()
    # inputs1 = inputs
    # print(inputs)
    outputs = model(inputs,inputs,modal=mode)
    outputs = outputs.data.cpu()
    return outputs


def _variance_to_confidence(variance, eps=1e-8):
    """Convert MC Dropout variance to confidence. Higher variance -> lower confidence."""
    # c = 1 / (1 + var), range (0, 1]
    var_scalar = variance.mean(dim=-1).clamp(min=eps)
    return (1.0 / (1.0 + var_scalar)).detach()


def extract_features(model, data_loader, print_freq=50, flip=True, mode=0, mc_drop=1):
    """
    Extract features with optional MC Dropout for uncertainty estimation.
    Returns:
        features: OrderedDict fname -> feature tensor (mean over MC samples)
        labels: OrderedDict fname -> pid
        confidences: OrderedDict fname -> scalar confidence in (0,1], or None if mc_drop<=1
    """
    model.eval()
    if mc_drop is not None and mc_drop > 1:
        _enable_dropout(model)
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()
    confidences = OrderedDict() if (mc_drop is not None and mc_drop > 1) else None

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            if mc_drop is not None and mc_drop > 1:
                mc_outputs = []
                mc_outputs_flip = []
                flip_imgs = fliplr(imgs)
                for _ in range(mc_drop):
                    out = extract_cnn_feature(model, imgs, mode)
                    out_flip = extract_cnn_feature(model, flip_imgs, mode)
                    mc_outputs.append(out)
                    mc_outputs_flip.append(out_flip)
                stacked = torch.stack(mc_outputs, dim=0)
                stacked_flip = torch.stack(mc_outputs_flip, dim=0)
                feat_combined = (stacked + stacked_flip) / 2.0  # [mc_drop, B, D]
                outputs = feat_combined.mean(0)
                variance = feat_combined.var(0)
                conf_batch = _variance_to_confidence(variance)
            else:
                outputs = extract_cnn_feature(model, imgs, mode)
                flip_imgs = fliplr(imgs)
                outputs_flip = extract_cnn_feature(model, flip_imgs, mode)
                conf_batch = None

            for idx, (fname, pid) in enumerate(zip(fnames, pids)):
                if conf_batch is None:
                    feat = (outputs[idx].detach() + outputs_flip[idx].detach()) / 2.0
                else:
                    feat = outputs[idx].detach()
                features[fname] = feat
                labels[fname] = pid
                if confidences is not None and conf_batch is not None:
                    c = conf_batch[idx]
                    confidences[fname] = c.item() if isinstance(c, torch.Tensor) else c

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, labels, confidences


def confidence_fusion_features(mu_v, mu_r, c_v, c_r, eps=1e-8):
    """
    Fuse visible (RGB) and infrared (IR) features using confidence and cross-modal similarity.
    w_v ∝ c_v · s_vr,  w_r ∝ c_r · s_vr
    f_fuse = (w_v * mu_v + w_r * mu_r) / (w_v + w_r)
    Args:
        mu_v: [D] or [N,D] visible feature mean
        mu_r: [D] or [N,D] infrared feature mean
        c_v: scalar or [N] visible confidence
        c_r: scalar or [N] infrared confidence
        eps: small constant for numerical stability
    Returns:
        f_fuse: fused feature
    """
    if isinstance(mu_v, torch.Tensor):
        pass
    else:
        mu_v = torch.tensor(mu_v, dtype=torch.float32)
        mu_r = torch.tensor(mu_r, dtype=torch.float32)
    need_squeeze = False
    if mu_v.dim() == 1:
        mu_v = mu_v.unsqueeze(0)
        mu_r = mu_r.unsqueeze(0)
        need_squeeze = True
    if not isinstance(c_v, torch.Tensor):
        c_v = torch.tensor(float(c_v), device=mu_v.device)
    if not isinstance(c_r, torch.Tensor):
        c_r = torch.tensor(float(c_r), device=mu_v.device)
    c_v = c_v.to(mu_v.device)
    c_r = c_r.to(mu_v.device)
    if c_v.dim() == 0:
        c_v = c_v.unsqueeze(0).expand(mu_v.size(0))
    if c_r.dim() == 0:
        c_r = c_r.unsqueeze(0).expand(mu_r.size(0))
    # s_vr: cosine similarity in [0,1] via (cos+1)/2
    mu_v_n = F.normalize(mu_v, p=2, dim=1)
    mu_r_n = F.normalize(mu_r, p=2, dim=1)
    cos_sim = (mu_v_n * mu_r_n).sum(dim=1)
    s_vr = ((cos_sim + 1) / 2).clamp(min=eps)
    w_v = c_v * s_vr
    w_r = c_r * s_vr
    w_sum = (w_v + w_r).clamp(min=eps)
    f_fuse = (w_v.unsqueeze(1) * mu_v + w_r.unsqueeze(1) * mu_r) / w_sum.unsqueeze(1)
    if need_squeeze:
        f_fuse = f_fuse.squeeze(0)
    return f_fuse


def pairwise_distance(features, query=None, gallery=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m, x.numpy(), y.numpy()


def evaluate_all(query_features, gallery_features, distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False,regdb=False):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams,regdb=regdb)
    print('Mean AP: {:4.1%}'.format(mAP))

    if (not cmc_flag):
        return mAP

    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True),}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams,regdb=regdb, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k-1]))
    return cmc_scores['market1501'], mAP


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, cmc_flag=False, rerank=False,modal=0,regdb=False):
        features, _, _ = extract_features(self.model, data_loader, mode=modal)
        # print(features,features) 
        distmat, query_features, gallery_features = pairwise_distance(features, query, gallery)
        # print(distmat)
        results = evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag,regdb=regdb)

        if (not rerank):
            return results

        print('Applying person re-ranking ...') 
        distmat_qq, _, _ = pairwise_distance(features, query, query)
        distmat_gg, _, _ = pairwise_distance(features, gallery, gallery)
        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
        return evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)
