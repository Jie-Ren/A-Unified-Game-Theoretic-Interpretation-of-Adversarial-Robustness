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
import torchvision.transforms as transforms
from torchvision import models
import matplotlib as mlt
mlt.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from tools.newlib import seed_torch, ImageNetSelectedDataset, normalize
from tools.attack import attack_magnitude_untarget
# robustness library
from robu.model_utils import make_and_restore_model
from robu import datasets as robustdatasets

'''
Generate adversarial images for untargetted attack
'''


def gene_advs(model, dataloader, args, save_dir, threshold):
    device = args.device if args.device == "cpu" else int(args.device)
    model.to(device)
    res = []
    count = 0  # num of correct classifications
    with torch.no_grad():
        model.eval()
        for i, (image_name, img, lbl) in enumerate(dataloader):
            print("img: %d" % i, image_name, 'label:', lbl)
            img_name = image_name[0].replace('.JPEG', '')
            img = img.to(device)
            lbl = lbl.to(device)

            ori_output, _ = model(normalize(img))
            pred_ori = torch.argmax(ori_output, dim=1)
            print('pred:', pred_ori.item())
            pred_logit = ori_output[:, pred_ori[0]]
            print('pred logit:', pred_logit)

            # untarget attack
            if pred_ori.item() == lbl.item():
                count += 1
                img_adv = attack_magnitude_untarget(model, img, lbl, pred_logit, device, step_size=args.step_size, epsilon=args.epsilon,
                                           perturb_steps=args.perturb_steps, threshold=threshold, distance=args.distance,
                                           isnormalize=True, mean=args.mean, std=args.std).to(device)

                output, _ = model(normalize(img_adv))
                pred_adv = torch.argmax(output, dim=1)
                out_gt = output[:, lbl[0]]
                dist = pred_logit - out_gt
                if dist > threshold and dist <= (threshold + 0.5):
                    print("Dist", dist)
                    delta = torch.abs(img_adv - img)
                    avg_delta = torch.sqrt(torch.norm(delta, p=2).pow(2) / (224 * 224 * 3))
                    img_adv = img_adv.detach().cpu().numpy()
                    np.save(save_dir + '/{}.npy'.format(img_name), img_adv)
                    tmp = "{}  label:{}  untarget label:{}  avg delta:{}".format(image_name[0], lbl.item(),
                                                                                 pred_adv.item(),
                                                                                 avg_delta.item())
                    res.append(tmp)
            else:
                print('Predict incorrectly.')
            print('----------------------------')

    print(count, 'images are classified correctly.')
    return res


def prepare(args):
    # prepare data-----------------
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

        dataset = ImageNetSelectedDataset(train=False, args=args, selected_num=args.base_num)
    else:
        print("No such dataset %s!" % args.dataset_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)  # bs=1

    # prepare model------------------
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
    parser.add_argument("--dataset_name", default="imagenet", type=str, help="dataset name, currently only support 'imagenet'")

    parser.add_argument('--base_num', default=100, type=int, help='# of images randomly sampled from the dataset')
    parser.add_argument('--distance', default='l_inf', type=str, help="type of adversarial attacks, currently only support 'l_inf'")
    parser.add_argument('--epsilon', default=32/255, type=float, help="epsilon for the PGD attack")
    parser.add_argument('--step_size', default=2/255, type=float, help="step_size for the PGD attack")
    parser.add_argument('--perturb_steps', default=100, type=int, help="perturb_steps for the PGD attack")
    parser.add_argument('--threshold', default=8.0, type=float, help="threshold of the attacking utility for the PGD attack")

    parser.add_argument("--adv_model", action="store_true", dest="adv_model", help="the type of model (True for adversarially learned DNN, False for standardly learned DNN)")
    parser.add_argument('--arch', default="resnet18", type=str, help="model name")
    parser.add_argument('--adv_model_path', type=str, default='pretrained_models/resnet18_linf_eps8.0.ckpt', help="model path of the adversarially learned DNN")
    parser.add_argument('--std_model_path', type=str, default='pretrained_models/resnet18_l2_eps0.ckpt', help="model path of the standardly learned DNN")
    parser.add_argument('--save_dir', default='advImgs_untarget', type=str, help="path to save the generated adversarial images")

    args = parser.parse_args()
    seed_torch(args.seed)

    # save path for generated adv_imgs based on ori_model or adv_model
    save_name = "adv_model" if args.adv_model else "ori_model"
    save_dir = os.path.join(args.distance, args.arch, "untarget", args.save_dir, save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save arguments
    with open(save_dir +'/result.txt', 'w') as f:
        f.writelines("decive:{}  seed:{}  base_num:{}  arch:{}\n".format(args.device, args.seed, args.base_num, args.arch))
        f.writelines("untarget attack: distance:{}   epsion:{}  step_size:{}  perturb_steps:{}  threshold:{}\n".format(args.distance, args.epsilon, args.step_size, args.perturb_steps, args.threshold))

    # main functions
    model, dataloader = prepare(args)
    res = gene_advs(model, dataloader, args, save_dir, args.threshold)

    with open(save_dir + '/result.txt', 'a') as f:
        for im in res:
            f.writelines(im+'\n')


if __name__ == "__main__":
    main()

