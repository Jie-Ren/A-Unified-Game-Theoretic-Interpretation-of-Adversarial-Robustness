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
# from tools.madrys import madrys
# robustness library
from robu.model_utils import make_and_restore_model
from robu import datasets as robustdatasets



def compute_order_interaction_img(img_name, lbl, args, save_dir, netout_dir):
    ori_interaction = []  # save order interactions of point-pairs in ori image
    adv_interaction = []  # save order interactions of point-pairs in adv image

    ori_logits = torch.load(os.path.join(netout_dir, '{}_output.pt'.format(img_name)))  # (pair_num*sample_num*4, 1000)
    adv_logits = torch.load(os.path.join(netout_dir, 'adv_{}_output.pt'.format(img_name)))

    ori_logits = ori_logits.reshape((args.pair_num, args.sample_num*4, 1000))
    adv_logits = adv_logits.reshape((args.pair_num, args.sample_num*4, 1000))
    tic = time.time()
    for i in range(args.pair_num):
        output_ori = ori_logits[i, :, :]
        output_adv = adv_logits[i, :, :]

        y_ori = F.log_softmax(output_ori, dim=1)[:, lbl[0]]
        y_adv = F.log_softmax(output_adv, dim=1)[:, lbl[0]]

        for k in range(args.sample_num):
            score_ori = y_ori[4 * k] + y_ori[4 * k + 3] - y_ori[4 * k + 1] - y_ori[4 * k + 2]
            score_adv = y_adv[4 * k] + y_adv[4 * k + 3] - y_adv[4 * k + 1] - y_adv[4 * k + 2]
            # cur_score = cur_score - inter_order_0
            ori_interaction.append(score_ori.item())
            adv_interaction.append(score_adv.item())

    ori_interaction = np.array(ori_interaction).reshape(-1, args.sample_num)
    adv_interaction = np.array(adv_interaction).reshape(-1, args.sample_num)
    assert ori_interaction.shape[0] == args.pair_num

    print('Image: %s, time: %.3f' % (img_name, time.time() - tic))
    print('--------------------------')
    tmp = "img{}_interaction.npy".format(img_name)
    np.save(os.path.join(save_dir, tmp), ori_interaction)  # (pair_num, sample_num)
    np.save(os.path.join(save_dir, "adv_" + tmp), adv_interaction)


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
            adv_output, _ = model(normalize(img_adv))
            target_lbl = torch.argmax(adv_output, dim=1)

            for r in args.ratios:
                print('Ratio:', r)
                save_name = "ratio{}".format(int(r * 100))
                res_dir = os.path.join('./', args.save_dir, save_name)
                netout_dir = os.path.join(args.netout_dir, save_name)
                if not os.path.exists(res_dir):
                    os.makedirs(res_dir)
                if args.out_type == 'GT':
                    compute_order_interaction_img(img_name, lbl, args, res_dir, netout_dir)
                elif args.out_type == 'Target':
                    compute_order_interaction_img(img_name, target_lbl, args, res_dir, netout_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/data/renjie/data/ImageNet', type=str, help="dataset path")
    parser.add_argument('--device', default=1, type=int, help="GPU ID")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--dataset_name", default="imagenet", type=str,
                        help="dataset name, currently only support 'imagenet'")

    parser.add_argument('--base_num', default=100, type=int, help='# of images randomly sampled from the dataset')
    parser.add_argument("--grid_size", default=16, type=int, help='number of grids of each img')

    parser.add_argument("--ratios", default=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95,1], type=list, help='ratios of context')
    parser.add_argument("--sample_num", default=100, type=int, help='sample num of S')
    parser.add_argument("--pair_num", default=200, type=int, help='number of point pair of each test img')

    parser.add_argument('--distance', default='l_inf', type=str, help="type of adversarial attacks, currently only support 'l_inf'")
    parser.add_argument("--targeted", action="store_true", dest="targeted", help="whether use the targeted attack (True for targeted attack, False for untargeted attack)")
    parser.add_argument('--out_type', default='GT', type=str, help="the type of the output used to compute interaction. 'GT' for the ground-truth label, 'Target' for the target label")  # GT; Target

    # for robustness library
    parser.add_argument("--adv_model", action="store_true", dest="adv_model", help="the type of model (True for adversarially learned DNN, False for standardly learned DNN)")
    parser.add_argument('--arch', default="resnet18", type=str, help="model name")
    parser.add_argument('--adv_model_path', type=str, default='pretrained_models/resnet18_linf_eps8.0.ckpt',
                        help="model path of the adversarially learned DNN")
    parser.add_argument('--std_model_path', type=str, default='pretrained_models/resnet18_l2_eps0.ckpt',
                        help="model path of the standardly learned DNN")

    args = parser.parse_args()
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
        model_type = 'adv_model'
        args.selected_imgs = os.listdir(os.path.join(prefix + args.img_adv, "adv_model"))
    else:
        print('Pytorch pretrained Model')
        model_type = 'ori_model'
        args.selected_imgs = os.listdir(os.path.join(prefix + args.img_adv, "ori_model"))


    args.save_dir = prefix + "saved_interactions_" + args.out_type + '/' + model_type
    args.adv_dir = prefix + args.img_adv + '/' + model_type
    args.netout_dir = prefix + "saved_logits" + '/' + model_type

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # save arguments
    with open(args.save_dir + '/result.txt', 'w') as f:
        f.writelines("decive:{}  seed:{}  base_num:{}  arch:{}\n".format(args.device, args.seed, args.base_num, args.arch))
        f.writelines("img_adv:{}\n".format(args.adv_dir))

    model, dataloader = prepare(args)
    compute_interactions(args, model, dataloader)


if __name__ == "__main__":
    main()
