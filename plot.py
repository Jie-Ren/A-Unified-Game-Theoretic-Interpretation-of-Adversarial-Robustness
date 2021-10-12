import sys
import numpy as np
import argparse
import os
import matplotlib as mlt
mlt.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def draw_hist(arch, orders, ori_mean,adv_mean, save_name):
    fig, ax = plt.subplots()
    bar_width = 0.02
    errorbar_config = {'capsize':3}
    ax.bar(orders, ori_mean, bar_width, yerr=None, error_kw=errorbar_config, label="ori_img")
    ax.bar(orders + bar_width, adv_mean, bar_width, yerr=None, error_kw=errorbar_config, label="adv_img")
    ax.set_xlabel("ratio of pixels")
    ax.set_ylabel("interaction")
    ax.set_title(arch)
    x = np.arange(0.1, 1.1, 0.1)
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])
    ax.legend()
    fig.tight_layout()
    plt.savefig(save_name)
    plt.close()


def draw_inter_diff(arch, orders, diff, save_name):
    fig, ax = plt.subplots()
    ax.plot(orders, diff, color='green')
    ax.fill_between(orders, 0, diff, facecolor='green', alpha=0.3)
    ax.set_xlabel('ratio of pixels')
    ax.set_ylabel("interaction difference")
    ax.set_title(arch)
    ax.set_xticks(np.arange(0.1, 1.1, 0.1))
    fig.tight_layout()
    plt.savefig(save_name)
    plt.close()


def draw_curve(arch, orders, ori_data, adv_data, save_name, ylabel='disentanglement'):
    plt.figure()
    plt.plot(orders, ori_data, label='ori_model')
    plt.plot(orders, adv_data, label='adv_model')
    plt.xlabel('ratio of pixels')
    plt.ylabel(ylabel)
    plt.title(arch)
    plt.legend()
    plt.xticks(np.arange(0.1, 1.1, 0.1))
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()

# all original imgs on ori_model and adv_model
def get_disentanglement(args, all_imgs):

    orders_disen = []
    adv_orders_disen = []
    for r in args.ratios:
        order = "ratio{}".format(int(r * 100))
        inters_dir = os.path.join(args.interaction_dir, "ori_model", order)
        inters_dir_advmodel = os.path.join(args.interaction_dir, "adv_model", order)
        r_inters_2, r_inters_3 = [], []
        adv_r_inters_2, adv_r_inters_3 = [], []
        for t in all_imgs:
            tmp = t + "_interaction.npy"
            inter = np.load(os.path.join(inters_dir, tmp))
            adv_inter = np.load(os.path.join(inters_dir_advmodel, tmp))

            inter_2 = np.abs(np.mean(inter, axis=1))
            inter_3 = np.mean(np.abs(inter), axis=1)
            r_inters_2.append(inter_2)
            r_inters_3.append(inter_3)

            adv_inter_2 = np.abs(np.mean(adv_inter, axis=1))
            adv_inter_3 = np.mean(np.abs(adv_inter), axis=1)
            adv_r_inters_2.append(adv_inter_2)
            adv_r_inters_3.append(adv_inter_3)

        r_inters_2 = np.array(r_inters_2)
        r_inters_3 = np.array(r_inters_3)
        adv_r_inters_2 = np.array(adv_r_inters_2)
        adv_r_inters_3 = np.array(adv_r_inters_3)
        disen_ori = np.mean(r_inters_2) / np.mean(r_inters_3)
        orders_disen.append(disen_ori)
        disen_adv = np.mean(adv_r_inters_2) / np.mean(adv_r_inters_3)
        adv_orders_disen.append(disen_adv)

    return np.array(orders_disen), np.array(adv_orders_disen)


def get_interaction_all(args, all_imgs, use_coef=False):

    orders_inter_mean = []
    orders_inter_std = []
    adv_orders_inter_mean = []
    adv_orders_inter_std = []
    for r in args.ratios:
        order = "ratio{}".format(int(r * 100))
        coef = 1
        if use_coef:
            # coefficient for m-order interaction
            tot_N = args.grid_size ** 2
            m = int((tot_N - 2) * r)  # m-order
            coef = (tot_N - 1 - m) * 1.0 / tot_N / (tot_N - 1)

        inters_dir = os.path.join(args.inter_dir, order)
        r_inters = []  # interactions for all imgs of this order
        adv_r_inters = []
        for t in all_imgs:
            tmp = t + "_interaction.npy"
            inter = np.load(os.path.join(inters_dir, tmp))  # (pair_num, sample_num)
            inter = np.mean(inter, axis=1)  # pair_num
            inter = coef * inter
            r_inters.append(inter)

            adv_inter = np.load(os.path.join(inters_dir, "adv_" + tmp))
            adv_inter = np.mean(adv_inter, axis=1)  # pair_num
            adv_inter = coef * adv_inter
            adv_r_inters.append(adv_inter)

        r_inters = np.array(r_inters)  # (imgs_num, pair_num)
        adv_r_inters = np.array(adv_r_inters)

        orders_inter_mean.append(np.mean(r_inters))
        orders_inter_std.append(np.std(r_inters, ddof=1))  # unbiased std
        adv_orders_inter_mean.append(np.mean(adv_r_inters))
        adv_orders_inter_std.append(np.std(adv_r_inters, ddof=1))

    return np.array(orders_inter_mean), np.array(orders_inter_std), np.array(adv_orders_inter_mean), np.array(adv_orders_inter_std)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratios", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1], type=list)
    parser.add_argument("--grid_size", default=16, type=int, help='number of grids of each img')

    parser.add_argument('--distance', default='l_inf', type=str,
                        help="type of adversarial attacks, currently only support 'l_inf'")
    parser.add_argument("--targeted", action="store_true", dest="targeted",
                        help="whether use the targeted attack (True for targeted attack, False for untargeted attack)")
    parser.add_argument('--out_type', default='GT', type=str,
                        help="the type of the output used to compute interaction. \n 'GT' for the ground-truth label \n 'Target' for the target label")  # GT; Target

    parser.add_argument("--adv_model", action="store_true", dest="adv_model",
                        help="the type of model (True for adversarially learned DNN, False for standardly learned DNN)")
    parser.add_argument('--arch', default="resnet18", type=str, help="model name")

    parser.add_argument('--type', default=1, type=int, help="1 for interactions; \n 2 for the difference in attacking utilities (weighted interaction); \n 3 for disentanglement")
    args = parser.parse_args()

    if not args.targeted:
        args.img_adv = 'advImgs_untarget'
        prefix = "{}/{}/untarget/".format(args.distance, args.arch)
    else:
        args.img_adv = 'advImgs_target'
        prefix = "{}/{}/target/".format(args.distance, args.arch)

    model_type = 'adv_model' if args.adv_model else 'ori_model'
    args.point_path = os.path.join(args.img_adv,model_type, "points")

    args.interaction_dir = prefix + "saved_interactions_" + args.out_type + '/'
    args.inter_dir = args.interaction_dir+model_type
    args.log_dir = os.path.join(args.interaction_dir.replace('saved_interactions', 'logs'))
    if args.type == 1 or args.type == 2:
        # compare interactions on ori_img vs. adv_img on the same model
        args.save_dir = os.path.join(args.log_dir, model_type, "type{}".format(args.type))
    else:
        # compare interactions on ori_model vs. adv_model on the same img
        args.save_dir = os.path.join(args.log_dir, "ori_imgs", "type{}".format(args.type))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    selected_imgs = os.listdir(os.path.join(prefix,args.point_path))

    orders = np.array(args.ratios)
    if args.type == 1:
        print('Draw raw interactions.')
        ori_inter_mean, ori_inter_std, adv_inter_mean, adv_inter_std = get_interaction_all(args, selected_imgs,                                                                     use_coef=False)
        # print('ori_mean std', ori_inter_mean, ori_inter_std)
        # print('adv_mean std', adv_inter_mean, adv_inter_std)
        draw_hist(args.arch, orders, ori_inter_mean,  adv_inter_mean, args.save_dir + '/all_interaction.png')

    elif args.type == 2:
        print('Draw difference in weighted interactions.')
        ori_inter_mean, _, adv_inter_mean, _ = get_interaction_all(args, selected_imgs, use_coef=True)
        draw_inter_diff(args.arch, orders, ori_inter_mean - adv_inter_mean, args.save_dir + '/all_diff.png')

    elif args.type == 3:
        print('Draw disentanglement.')
        ori_disentangle, adv_disentangle = get_disentanglement(args, selected_imgs)
        # print('ori model', ori_disentangle)
        # print('adv model', adv_disentangle)
        draw_curve(args.arch, orders, ori_disentangle, adv_disentangle, args.save_dir + '/all_disentangle.png', 'disentangle')

























