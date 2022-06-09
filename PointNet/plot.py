import sys
import numpy as np
import argparse
import os
import matplotlib as mlt
mlt.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def draw_hist(orders, ori_mean, adv_mean, save_name):
    fig, ax = plt.subplots()
    bar_width = 0.02
    errorbar_config = {'capsize':3}
    ax.bar(orders, ori_mean, bar_width, yerr=None, error_kw=errorbar_config, label="ori_img")
    ax.bar(orders + bar_width, adv_mean, bar_width, yerr=None, error_kw=errorbar_config, label="adv_img")
    ax.set_xlabel("ratio of pixels")
    ax.set_ylabel("interaction")
    x = np.arange(0.1, 1.1, 0.1)
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])
    ax.legend()
    fig.tight_layout()
    plt.savefig(save_name)
    plt.close()



def get_interaction_all(args, all_imgs):

    orders_inter_mean = []
    adv_orders_inter_mean = []

    for r in args.ratios:
        order = "ratio{}".format(int(r * 100))

        inters_dir = os.path.join(args.interaction_dir, order)
        r_inters = []  # interactions for all imgs of this order
        adv_r_inters = []
        for t in all_imgs:
            tmp = t + "_interaction.npy"
            inter = np.load(os.path.join(inters_dir, tmp))  # (pair_num, sample_num)
            inter = np.mean(inter, axis=1)  # pair_num
            r_inters.append(inter)

            adv_inter = np.load(os.path.join(inters_dir, "adv_" + tmp))
            adv_inter = np.mean(adv_inter, axis=1)  # pair_num
            adv_r_inters.append(adv_inter)

        r_inters = np.array(r_inters)  # (imgs_num, pair_num)
        adv_r_inters = np.array(adv_r_inters)

        orders_inter_mean.append(np.mean(r_inters))
        adv_orders_inter_mean.append(np.mean(adv_r_inters))

    return np.array(orders_inter_mean), np.array(adv_orders_inter_mean)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratios", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1], type=list)
    parser.add_argument("--grid_size", default=32, type=int, help='number of regions of each sample')

    parser.add_argument('--arch', type=str, default='pointnet2')
    parser.add_argument('--dataset', type=str, default='modelnet40')

    args = parser.parse_args()

    prefix = "{}-{}".format(args.arch, args.dataset)

    args.point_path = os.path.join(prefix, "points")
    args.interaction_dir = os.path.join(prefix, "saved_interactions")
    args.save_dir = os.path.join(prefix, "figs")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    selected_imgs = os.listdir(os.path.join(args.point_path))

    orders = np.array(args.ratios)
    print('Draw raw interactions.')
    ori_inter_mean, adv_inter_mean = get_interaction_all(args, selected_imgs)
    print(ori_inter_mean,adv_inter_mean)
    draw_hist(orders, ori_inter_mean,  adv_inter_mean, args.save_dir + '/all_interaction.png')