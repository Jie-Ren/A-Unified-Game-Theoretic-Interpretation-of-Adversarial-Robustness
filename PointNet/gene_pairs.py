import sys
import os
import argparse
import time
import copy
import numpy as np
from torch.utils.data import DataLoader
from data_utils import *
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--root', type=str, default='data/modelnet40_numpy', help='data path')
    parser.add_argument('--arch', type=str, default='pointnet2')
    parser.add_argument('--dataset', type=str, default='modelnet40')
    parser.add_argument('--num_points', type=int, default=1024, help='num of points to use')

    parser.add_argument("--ratios", default=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95,1], type=list,
                        help='ratios of context')
    parser.add_argument("--sample_num", default=100, type=int, help='sample num of S')
    parser.add_argument("--grid_size", default=32, type=int, help='number of regions in each sample')
    parser.add_argument("--pair_num", default=50, type=int, help='number of point pair of each test sample')
    parser.add_argument('--img_adv', default='advImgs_untarget', type=str, help="path for generated adv images")

    args = parser.parse_args()
    np.random.seed(args.seed)

    prefix = os.path.join("{}-{}".format(args.arch,args.dataset))
    args.selected_imgs = os.listdir(os.path.join(prefix, args.img_adv))
    args.selected_imgs.sort()

    args.point_dir = os.path.join(prefix, "points")
    if not os.path.exists(args.point_dir):
        os.makedirs(args.point_dir)

    test_loader = DataLoader(ModelNet_Loader(args, partition='train', num_points=args.num_points), num_workers=8,
                             batch_size=1, shuffle=False, drop_last=False)

    # need to first divide the input into 32 regions, and compute the center of each region
    fps_indices = np.load('fps_1024_32_index.npy')  # (100, 32) center of the regions, a total of 100 samples
    fps_indices = torch.from_numpy(fps_indices)

    for i, (image_name, img, lbl) in enumerate(test_loader):
        img_name = image_name[0]
        if not (img_name + '.npy' in args.selected_imgs):
            continue
        print("img: %d" % i, img_name, 'label:', lbl)
        # img: (B, 1024, 3)
        fps_index = fps_indices[i]  # (32,)
        data_fps = img[:, fps_index, :]  # (B, 32, 3), centroids of each region
        data_fps = data_fps.squeeze(0)

        save_path = os.path.join(args.point_dir, "{}".format(img_name))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

            tot_pairs = []
            for k in range(args.pair_num):
                while True:
                    x1 = np.random.randint(0, args.grid_size)
                    t = np.random.choice([1, 2, 3], 1)  # 距离最近的3个region中选1个
                    cur_region = data_fps[x1]
                    region_dist = torch.norm(data_fps - cur_region, p=2, dim=1)
                    sorted, idxes = torch.sort(region_dist)
                    x2 = idxes[t]
                    if [x1, x2] in tot_pairs or [x2, x1] in tot_pairs:
                        continue
                    else:
                        tot_pairs.append(list([x1, x2]))
                        break
            tot_pairs = np.array(tot_pairs)
            np.save(save_path + '/points.npy', tot_pairs)

            for r in args.ratios:
                print('Ratio:', r)
                players = []
                for p, pt in enumerate(tot_pairs):
                    point1, point2 = pt[0], pt[1]
                    # m-order interactions
                    context = list(range(args.grid_size))
                    context.remove(point1)
                    context.remove(point2)

                    players_thispair = []
                    m = int((args.grid_size - 2) * r)  # m-order
                    for k in range(args.sample_num):
                        players_thispair.append(np.random.choice(context, m, replace=False))

                    players.append(players_thispair)

                player_save_path = os.path.join(args.point_dir, "{}/ratio{}_S.npy".format(img_name, int(r * 100)))
                players = np.array(players)
                np.save(player_save_path, players)


if __name__ == "__main__":
    main()
