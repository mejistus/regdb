from __future__ import print_function, absolute_import

import os
import os.path as osp
import random
import time

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torch.autograd import Variable


class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size=(144, 288)):
        test_image = []
        for img_file in test_img_file:
            img = Image.open(img_file)
            img = img.resize((img_size[0], img_size[1]), Image.LANCZOS)
            test_image.append(np.array(img))
        self.test_image = np.array(test_image)
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.test_image[index], self.test_label[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.test_image)


def fliplr(img):
    """Flip a BCHW tensor horizontally."""
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
    return img.index_select(3, inv_idx)


def extract_reid_features(model, loader, num_images, mode, feature_name, pool_dim=2048):
    net = model
    net.eval()
    print('Extracting {} Feature...'.format(feature_name))
    start = time.time()
    ptr = 0
    features = np.zeros((num_images, pool_dim))
    with torch.no_grad():
        for inputs, _ in loader:
            batch_num = inputs.size(0)
            flip_inputs = fliplr(inputs)
            inputs = Variable(inputs.cuda())
            feat = net(inputs, inputs, mode)
            flip_inputs = Variable(flip_inputs.cuda())
            flip_feat = net(flip_inputs, flip_inputs, mode)
            feature = (feat.detach() + flip_feat.detach()) / 2
            fnorm = torch.norm(feature, p=2, dim=1, keepdim=True)
            feature = feature.div(fnorm.expand_as(feature))
            features[ptr:ptr + batch_num, :] = feature.cpu().numpy()
            ptr += batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return features


def extract_regdb_gallery_features(model, gall_loader, ngall):
    return extract_reid_features(model, gall_loader, ngall, mode=2, feature_name='Gallery')


def extract_regdb_query_features(model, query_loader, nquery):
    return extract_reid_features(model, query_loader, nquery, mode=1, feature_name='Query')


def extract_sysu_gallery_features(model, gall_loader, ngall):
    return extract_reid_features(model, gall_loader, ngall, mode=1, feature_name='Gallery')


def extract_sysu_query_features(model, query_loader, nquery):
    return extract_reid_features(model, query_loader, nquery, mode=2, feature_name='Query')


def pairwise_distance(features_q, features_g):
    x = torch.from_numpy(features_q)
    y = torch.from_numpy(features_g)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m.numpy()


def process_test_regdb(img_dir, trial=1, modal='visible'):
    if modal == 'visible':
        input_data_path = osp.join(img_dir, 'idx', 'test_visible_{}.txt'.format(trial))
    elif modal == 'thermal':
        input_data_path = osp.join(img_dir, 'idx', 'test_thermal_{}.txt'.format(trial))
    else:
        raise ValueError("Unsupported RegDB modal: {}".format(modal))

    with open(input_data_path, 'rt') as f:
        data_file_list = f.read().splitlines()
    file_image = [osp.join(img_dir, s.split(' ')[0]) for s in data_file_list]
    file_label = [int(s.split(' ')[1]) for s in data_file_list]
    return file_image, np.array(file_label)


def process_query_sysu(data_path, mode='all', relabel=False):
    if mode not in ('all', 'indoor'):
        raise ValueError("Unsupported SYSU mode: {}".format(mode))
    ir_cameras = ['cam3', 'cam6']

    file_path = os.path.join(data_path, 'exp/test_id.txt')
    files_ir = []
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for pid in sorted(ids):
        for cam in ir_cameras:
            img_dir = os.path.join(data_path, cam, pid)
            if os.path.isdir(img_dir):
                files_ir.extend(sorted([img_dir + '/' + i for i in os.listdir(img_dir)]))

    query_img = []
    query_id = []
    query_cam = []
    for img_path in files_ir:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)
    return query_img, np.array(query_id), np.array(query_cam)


def process_gallery_sysu(data_path, mode='all', trial=0, relabel=False):
    random.seed(trial)
    if mode == 'all':
        rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
    elif mode == 'indoor':
        rgb_cameras = ['cam1', 'cam2']
    else:
        raise ValueError("Unsupported SYSU mode: {}".format(mode))

    file_path = os.path.join(data_path, 'exp/test_id.txt')
    files_rgb = []
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for pid in sorted(ids):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path, cam, pid)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_rgb.append(random.choice(new_files))

    gall_img = []
    gall_id = []
    gall_cam = []
    for img_path in files_rgb:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        gall_img.append(img_path)
        gall_id.append(pid)
        gall_cam.append(camid)
    return gall_img, np.array(gall_id), np.array(gall_cam)


def eval_regdb(distmat, q_pids, g_pids, max_rank=20):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.
    q_camids = np.ones(num_q).astype(np.int32)
    g_camids = 2 * np.ones(num_g).astype(np.int32)

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        raw_cmc = matches[q_idx][keep]
        if not np.any(raw_cmc):
            continue

        cmc = raw_cmc.cumsum()
        pos_idx = np.where(raw_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx] / (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        all_AP.append(tmp_cmc.sum() / num_rel)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return all_cmc, mAP, mINP


def eval_sysu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    pred_label = g_pids[indices]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    new_all_cmc = []
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = indices[q_idx]
        remove = (q_camid == 3) & (g_camids[order] == 2)
        keep = np.invert(remove)

        new_cmc = pred_label[q_idx][keep]
        new_index = np.unique(new_cmc, return_index=True)[1]
        new_cmc = [new_cmc[index] for index in sorted(new_index)]

        new_match = (new_cmc == q_pid).astype(np.int32)
        new_cmc = new_match.cumsum()
        new_all_cmc.append(new_cmc[:max_rank])

        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue

        cmc = orig_cmc.cumsum()
        pos_idx = np.where(orig_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx] / (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        all_AP.append(tmp_cmc.sum() / num_rel)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    new_all_cmc = np.asarray(new_all_cmc).astype(np.float32)
    new_all_cmc = new_all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return new_all_cmc, mAP, mINP
