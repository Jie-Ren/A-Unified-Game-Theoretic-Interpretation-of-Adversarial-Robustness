import sys
import sys, os
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
from PIL import Image
from torchvision import transforms
from torchvision import models

from tools.newlib import seed_torch, ImageNetSelectedDataset, normalize
# robustness library
from robu.model_utils import make_and_restore_model
from robu import datasets as robustdatasets


def compute_order_interaction_img(model, img, img_adv, img_name, point_list, args, save_dir, r):
    device = 0  # Do not modify
    model.to(device)

    with torch.no_grad():
        model.eval()
        img_size = args.img_size
        channels = img.size(1)

        tic = time.time()
        players = np.load(os.path.join(args.point_path, "img{}/ratio{}_S.npy".format(img_name, int(r * 100))))
        ori_logits = []
        adv_logits = []

        forward_mask = []
        for p, pt in enumerate(point_list):
            point1, point2 = pt[0], pt[1]

            players_thispair = players[p]
            m = int((args.grid_size ** 2 - 2) * r)  # m-order
            mask = torch.zeros((4 * args.sample_num, channels, args.grid_size * args.grid_size)).to(device)

            for k in range(args.sample_num):
                mask[4*k:4*(k+1), :, players_thispair[k]] = 1  # S
                mask[4*k+1, :, point1] = 1  # S U {i}
                mask[4*k+2, :, point2] = 1  # S U {j}
                mask[4*k, :, point1] = 1
                mask[4*k, :, point2] = 1    # S U {i,j}
            mask = mask.view(4 * args.sample_num, channels, args.grid_size, args.grid_size)
            mask = F.interpolate(mask.clone(), size=[img_size, img_size], mode="nearest").float()

            forward_mask.append(mask)
            if (len(forward_mask) < args.cal_batch // args.sample_num) and (p < args.pair_num - 1):
                continue
            else:
                forward_batch = len(forward_mask) * args.sample_num
                batch_mask = torch.cat(forward_mask, dim=0)
                expand_img = img.expand(4 * forward_batch, -1, img_size, img_size).clone()
                expand_img_adv = img_adv.expand(4 * forward_batch, -1, img_size, img_size).clone()
                masked_img = batch_mask * expand_img
                masked_img_adv = batch_mask * expand_img_adv

                output_ori, _ = model(masked_img)  # (cal_batch*4, 1000)
                output_adv, _ = model(masked_img_adv)

                ori_logits.append(output_ori.detach())
                adv_logits.append(output_adv.detach())
                forward_mask = []

        all_ori_logits = torch.cat(ori_logits, dim=0)  # (pair_num*sample_num*4, 1000)
        all_adv_logits = torch.cat(adv_logits, dim=0)

        print('Image: %s, time: %.3f' % (img_name, time.time() - tic))
        print('--------------------------')
        tmp = "{}_output.pt".format(img_name)
        torch.save(all_ori_logits, os.path.join(save_dir, tmp))
        torch.save(all_adv_logits, os.path.join(save_dir, "adv_" + tmp))



def compute_interactions(args, model, dataloader):
    device = 0  # Do not modify
    model.to(device)

    with torch.no_grad():
        model.eval()
        for i, (image_name, img, lbl) in enumerate(dataloader):
            img_name = image_name[0].replace('.JPEG', '')
            if not (img_name + '.npy' in args.selected_imgs):
                continue
            print("img: %d" % i, image_name, 'label:', lbl)

            img = img.to(device)
            lbl = lbl.to(device)

            img_adv = np.load(os.path.join(args.adv_dir, "{}.npy".format(img_name)))  # ours
            img_adv = torch.from_numpy(img_adv).to(device)

            img = normalize(img)
            img_adv = normalize(img_adv)

            points_for_img = os.path.join(args.point_path, "img{}".format(img_name), 'points.npy')
            point_list = np.load(points_for_img)

            for r in args.ratios:
                print('Ratio:', r)
                save_name = "ratio{}".format(int(r * 100))
                res_dir = os.path.join('./', args.save_dir, save_name)
                if not os.path.exists(res_dir):
                    os.makedirs(res_dir)
                compute_order_interaction_img(model, img, img_adv, img_name, point_list, args, res_dir, r)

def prepare(args):
    # prepare data---------------
    # Only for ImageNet
    if args.dataset_name == "imagenet":
        class_num = 1000
        args.img_size = 224
        args.mean = [0.485, 0.456, 0.406]
        args.std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        args.transform = transform
        # need args.root, args.img_size, args.seed
        dataset = ImageNetSelectedDataset(train=False, args=args, selected_num=args.base_num)
    else:
        print("No such dataset %s!" % args.dataset_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)  # bs=1

    # prepare models----------------
    if args.adv_model:
        print('Load adversarial model.')
        ds = robustdatasets.ImageNet(args.root)
        model, _ = make_and_restore_model(arch=args.arch, dataset=ds,
                                          resume_path=args.adv_model_path,
                                          pytorch_pretrained=False)
    else:
        print('Load pytorch pretrained model.')
        ds = robustdatasets.ImageNet(args.root)
        model, _ = make_and_restore_model(arch=args.arch, dataset=ds,
                                          resume_path=args.std_model_path,
                                          pytorch_pretrained=False)

    return model, dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/data/renjie/data/ImageNet', type=str, help="dataset path")
    parser.add_argument('--device', default=1, type=int, help="GPU ID")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--grid_size", default=16, type=int, help='number of grids of each img')
    parser.add_argument("--dataset_name", default="imagenet", type=str, help="dataset name, currently only support 'imagenet'")
    parser.add_argument('--base_num', default=100, type=int, help='# of images randomly sampled from the dataset')

    parser.add_argument("--ratios", default=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95,1], type=list, help='ratios of context')
    parser.add_argument("--sample_num", default=100, type=int, help='sample num of S')
    parser.add_argument("--pair_num", default=200, type=int, help='number of point pair of each test img')
    # cal_batch should be divisible by sample_num
    parser.add_argument("--cal_batch", default=100, type=int, help='calculate # of images per batch')
    # for robustness library
    parser.add_argument("--targeted", action="store_true", dest="targeted", help="whether use the targeted attack")
    parser.add_argument("--adv_model", action="store_true", dest="adv_model", help="the type of model (True for adversarially learned DNN, False for standardly learned DNN)")
    parser.add_argument('--arch', default="resnet18", type=str)
    parser.add_argument('--distance', default='l_inf', type=str)
    parser.add_argument('--adv_model_path', type=str, default='pretrained_models/resnet18_linf_eps8.0.ckpt',
                        help="model path of the adversarially learned DNN")
    parser.add_argument('--std_model_path', type=str, default='pretrained_models/resnet18_l2_eps0.ckpt',
                        help="model path of the standardly learned DNN")


    args = parser.parse_args()
    args.targetet=True
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    seed_torch(args.seed)

    if not args.targeted:
        args.img_adv = 'advImgs_untarget'
        prefix = "{}/{}/untarget/".format(args.distance, args.arch)
    else:
        args.img_adv = 'advImgs_target'
        prefix = "{}/{}/target/".format(args.distance, args.arch)


    if args.adv_model:
        print('Adversarial Model.')
        args.save_dir = prefix + "saved_logits" + '/adv_model/'
        args.adv_dir = prefix + args.img_adv + '/adv_model/'
        args.selected_imgs = os.listdir(os.path.join(prefix + args.img_adv, "adv_model"))
        args.point_path = os.path.join(args.img_adv, "adv_model", "points")
    else:
        print('Pytorch pretrained Model')
        args.save_dir = prefix + "saved_logits" + '/ori_model/'
        args.adv_dir = prefix + args.img_adv + '/ori_model/'
        args.selected_imgs = os.listdir(os.path.join(prefix + args.img_adv, "ori_model"))
        args.point_path = os.path.join(args.img_adv, "ori_model", "points")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args.point_path = os.path.join(prefix, args.point_path)
    if not os.path.exists(args.point_path):
        os.makedirs(args.point_path)

    # save arguments
    with open(args.save_dir + 'result.txt', 'w') as f:
        f.writelines("decive:{}  seed:{}  base_num:{}  arch:{}  cal_batch:{}\n".format(args.device, args.seed, args.base_num, args.arch, args.cal_batch))
        f.writelines("img_adv:{}  point_path:{}\n".format(args.adv_dir, args.point_path))

    model, dataloader = prepare(args)
    compute_interactions(args, model, dataloader)


if __name__ == "__main__":
    main()
