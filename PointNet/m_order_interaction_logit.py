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
import random
from torch.utils.data import DataLoader
from collections import OrderedDict

from models.pointnet2 import PointNet2ClsMsg
from data_utils import *


def prepare(args):
    test_loader = DataLoader(ModelNet_Loader(args, partition='train', num_points=args.num_points), num_workers=8,
                             batch_size=1, shuffle=False, drop_last=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.arch == 'pointnet2':
        args.k = 20
        model = PointNet2ClsMsg(args).to(device)
    else:
        raise ValueError("The network architecture is not supported!")
    state_dict = torch.load(args.model_path, map_location=device)
    new_state_dict = OrderedDict()

    # only for model trained on 2 gpus
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model = model.eval()
    return model, test_loader


def compute_order_interaction_img(model, region_id, center, img, img_adv, img_name, point_list, args, save_dir, r):
    device = 0
    model.to(device)

    with torch.no_grad():
        model.eval()
        N = img.size(1)  # (B, 1024, 3)

        tic = time.time()
        players = np.load(os.path.join(args.point_path, "{}/ratio{}_S.npy".format(img_name, int(r * 100))))
        ori_logits = []
        adv_logits = []

        for p, pt in enumerate(point_list):
            region1, region2 = pt[0], pt[1]

            players_thispair = players[p]
            mask = torch.zeros((4 * args.sample_num, N, 3)).to(device)

            for k in range(args.sample_num):
                for h in players_thispair[k]:
                    mask[4 * k:4 * (k + 1), region_id == h, :] = 1  # S

                mask[4 * k + 1, region_id == region1, :] = 1  # S U {i}
                mask[4 * k + 2, region_id == region2, :] = 1  # S U {j}
                mask[4 * k, region_id == region1, :] = 1
                mask[4 * k, region_id == region2, :] = 1  # S U {i,j}

            batch_num = int(args.sample_num / args.cal_batch)
            if args.sample_num % args.cal_batch > 0:
                batch_num += 1
            for batch in range(batch_num):
                st = batch * args.cal_batch
                ed = (batch + 1) * args.cal_batch
                if ed > args.sample_num:
                    ed = args.sample_num
                cnt = ed -st
                expand_img = img.expand(4*cnt, N, 3).clone()
                expand_img_adv = img_adv.expand(4*cnt, N, 3).clone()
                masked_img = mask[(4*st):(4*ed)] * expand_img
                masked_img_adv = mask[(4*st):(4*ed)] * expand_img_adv

                output_ori = model(masked_img.permute(0, 2, 1))    # (cnt*4, 10)
                output_adv = model(masked_img_adv.permute(0, 2, 1))

                ori_logits.append(output_ori.detach())
                adv_logits.append(output_adv.detach())

        all_ori_logits = torch.cat(ori_logits, dim=0)  # (pair_num*sample_num*4, 10)
        all_adv_logits = torch.cat(adv_logits, dim=0)
        assert all_ori_logits.shape[0] == args.pair_num*args.sample_num*4

        print('Image: %s, time: %.3f' % (img_name, time.time() - tic))
        print('--------------------------')
        tmp = "{}_output.pt".format(img_name)
        torch.save(all_ori_logits, os.path.join(save_dir, tmp))
        torch.save(all_adv_logits, os.path.join(save_dir, "adv_" + tmp))


def cal_region_id(data, fps_index):
    data_fps = data[:, fps_index, :]  # (B, 32, 3), centroids of each region
    distance = square_distance(data, data_fps)  # (B, 1024, 32), here B=1
    region_id = torch.argmin(distance, dim=2)  # (B, 1024), B=1, ids for each region
    return region_id.squeeze()


def compute_interactions(args, model, dataloader):
    device = 0  # Do not modify
    model.to(device)

    with torch.no_grad():
        fps_indices = np.load('fps_1024_32_index.npy')  # (100, 32) center of the regions
        fps_indices = torch.from_numpy(fps_indices)

        for i, (image_name, img, lbl) in enumerate(dataloader):
            img_name = image_name[0]
            if not (img_name+'.npy' in args.selected_imgs):
                continue
            print("img: %d" % i, img_name, 'label:', lbl)
            img = img.to(device)  # (B, 1024, 3)
            lbl = lbl.to(device)  # (B,)

            fps_index = fps_indices[i]  # (32,)
            region_id = cal_region_id(img, fps_index)  # (1024,)
            center = torch.mean(img, dim=1).squeeze()  # (3,)  # mean value of all points in a sample

            img_adv = np.load(os.path.join(args.adv_dir, "{}.npy".format(img_name)))
            img_adv = torch.from_numpy(img_adv).to(device)  # (B, 3, 1024)
            img_adv = img_adv.permute(0, 2, 1)

            points_for_img = os.path.join(args.point_path, "{}".format(img_name), 'points.npy')
            point_list = np.load(points_for_img)

            for r in args.ratios:
                print('Ratio:', r)
                save_name = "ratio{}".format(int(r * 100))
                res_dir = os.path.join('./', args.save_dir, save_name)
                if not os.path.exists(res_dir):
                    os.makedirs(res_dir)
                compute_order_interaction_img(model, region_id, center, img, img_adv, img_name, point_list, args, res_dir, r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=1, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument('--root', type=str, default='data/modelnet40_numpy', help='data path')
    parser.add_argument('--arch', type=str, default='pointnet2')
    parser.add_argument('--dataset', type=str, default='modelnet40')
    parser.add_argument('--num_points', type=int, default=1024, help='num of points to use')

    parser.add_argument("--grid_size", default=32, type=int, help='number of regions in each sample')
    parser.add_argument("--ratios", default=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95,1], type=list, help='ratios of context')
    parser.add_argument("--sample_num", default=100, type=int, help='sample num of S')
    parser.add_argument("--pair_num", default=50, type=int, help='number of point pair of each test sample')
    # cal_batch should be divisible by sample_num
    parser.add_argument("--cal_batch", default=25, type=int, help='calculate # of samples per forward')

    parser.add_argument('--model_path', default="model_best.t7", type=str, help='model path')
    parser.add_argument('--img_adv', default='advImgs_untarget', type=str, help="path for generated adv samples")
    parser.add_argument('--save_dir', default='saved_logits', type=str)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    set_random(args.seed)

    prefix = os.path.join("{}-{}".format(args.arch,args.dataset))
    args.adv_dir = os.path.join(prefix, args.img_adv)
    args.selected_imgs = os.listdir(os.path.join(prefix, args.img_adv))
    args.point_path = os.path.join(prefix, "points")
    args.save_dir = os.path.join(prefix, args.save_dir)

    model, dataloader = prepare(args)
    compute_interactions(args, model, dataloader)


if __name__ == '__main__':
    main()