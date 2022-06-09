import sys
import os
import argparse
import time
import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
from collections import OrderedDict

from data_utils import *


def compute_order_interaction_img(img_name, lbl, args, save_dir, netout_dir):
    ori_interaction = []  # save order interactions of point-pairs in ori image
    adv_interaction = []  # save order interactions of point-pairs in adv image

    ori_logits = torch.load(os.path.join(netout_dir, '{}_output.pt'.format(img_name)))  # (pair_num*sample_num*4, 10)
    adv_logits = torch.load(os.path.join(netout_dir, 'adv_{}_output.pt'.format(img_name)))

    ori_logits = ori_logits.reshape((args.pair_num, args.sample_num * 4, 10))
    adv_logits = adv_logits.reshape((args.pair_num, args.sample_num * 4, 10))
    tic = time.time()
    for i in range(args.pair_num):
        output_ori = ori_logits[i, :, :]
        output_adv = adv_logits[i, :, :]
        y_ori = F.log_softmax(output_ori, dim=1)[:, lbl[0]]
        y_adv = F.log_softmax(output_adv, dim=1)[:, lbl[0]]

        for k in range(args.sample_num):
            score_ori = y_ori[4 * k] + y_ori[4 * k + 3] - y_ori[4 * k + 1] - y_ori[4 * k + 2]
            score_adv = y_adv[4 * k] + y_adv[4 * k + 3] - y_adv[4 * k + 1] - y_adv[4 * k + 2]
            ori_interaction.append(score_ori.item())
            adv_interaction.append(score_adv.item())

    ori_interaction = np.array(ori_interaction).reshape(-1, args.sample_num)
    adv_interaction = np.array(adv_interaction).reshape(-1, args.sample_num)
    assert ori_interaction.shape[0] == args.pair_num

    print('Image: %s, time: %.3f' % (img_name, time.time() - tic))
    print('--------------------------')
    tmp = "{}_interaction.npy".format(img_name)
    np.save(os.path.join(save_dir, tmp), ori_interaction)  # (pair_num, sample_num)
    np.save(os.path.join(save_dir, "adv_" + tmp), adv_interaction)


def compute_interactions(args,  dataloader):
    device = 0  # Do not modify

    with torch.no_grad():
        for i, (image_name, img, lbl) in enumerate(dataloader):
            img_name = image_name[0]
            if not (img_name+'.npy' in args.selected_imgs):
                continue

            img = img.to(device)  # (B, 1024, 3)
            lbl = lbl.to(device)  # (B,)

            for r in args.ratios:
                print('Ratio:', r)
                save_name = "ratio{}".format(int(r * 100))
                res_dir = os.path.join('./', args.save_dir, save_name)
                netout_dir = os.path.join(args.netout_dir, save_name)
                if not os.path.exists(res_dir):
                    os.makedirs(res_dir)
                compute_order_interaction_img(img_name, lbl, args, res_dir, netout_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--root', type=str, default='data/modelnet40_numpy', help='data path')
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument('--arch', type=str, default='pointnet2')
    parser.add_argument('--dataset', type=str, default='modelnet40')
    parser.add_argument('--num_points', type=int, default=1024, help='num of points to use')

    parser.add_argument("--grid_size", default=32, type=int, help='number of regions of each sample')
    parser.add_argument("--ratios", default=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95,1], type=list, help='ratios of context')
    parser.add_argument("--sample_num", default=100, type=int, help='sample num of S')
    parser.add_argument("--pair_num", default=50, type=int, help='number of point pair of each test sample')
    # cal_batch should be divisible by sample_num
    parser.add_argument("--cal_batch", default=25, type=int, help='calculate # of samples per forward')

    parser.add_argument('--img_adv', default='advImgs_untarget', type=str, help="path for generated adv samples")
    parser.add_argument('--save_dir', default='saved_interactions', type=str)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    set_random(args.seed)

    prefix = os.path.join("{}-{}".format(args.arch,args.dataset))
    args.selected_imgs = os.listdir(os.path.join(prefix, args.img_adv))
    args.point_path = os.path.join(prefix, "points")
    args.netout_dir = os.path.join(prefix, "saved_logits")
    args.save_dir = os.path.join(prefix, args.save_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataloader = DataLoader(ModelNet_Loader(args, partition='train', num_points=args.num_points), num_workers=8,
                             batch_size=1, shuffle=False, drop_last=False)
    compute_interactions(args, dataloader)


if __name__ == '__main__':
    main()
