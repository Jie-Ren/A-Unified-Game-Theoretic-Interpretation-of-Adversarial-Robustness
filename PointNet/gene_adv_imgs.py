import os, sys
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import random

from models.pointnet2 import PointNet2ClsMsg
from data_utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = str(0)  # change GPU here


def set_random(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare(args):
    test_loader = DataLoader(ModelNet_Loader(args, partition='train', num_points=args.num_points), num_workers=8,
                             batch_size=1, shuffle=True, drop_last=False)
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


def attack_magnitude_untarget(model,
              x_natural,
              y,
              ori_logit,
              device,
              step_size=2/255,
              epsilon=16/255,
              perturb_steps=100,
              threshold=4.0,
              distance='l_inf'
           ):
    model.eval()
    if distance == 'l_inf':
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()    # random start
    elif distance == 'l_2':
        delta = torch.zeros_like(x_natural).to(device).detach()  # (B,3,1024)
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
        x_adv = x_natural.detach() + delta
    x_adv = torch.clamp(x_adv, min=-1.0, max=1.0)

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            output = model(x_adv)
            loss_ce = F.cross_entropy(output, y)

        grad = torch.autograd.grad(loss_ce, [x_adv])[0]
        is_valid = False
        while not is_valid:
            if distance == 'l_inf':
                x_adv_tmp = x_adv.detach() + step_size * torch.sign(grad.detach())
                x_adv_tmp = torch.min(torch.max(x_adv_tmp, x_natural - epsilon), x_natural + epsilon)
                x_adv_tmp = torch.clamp(x_adv_tmp, -1.0, 1.0)
            elif distance == 'l_2':
                g_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, 1, 1)
                scaled_g = grad / (g_norm + 1e-10)
                x_adv_tmp = x_adv.detach() + step_size * scaled_g
                delta = x_adv_tmp - x_natural
                delta = delta.renorm(p=2, dim=0, maxnorm=epsilon)
                x_adv_tmp = torch.clamp(x_natural + delta, -1.0, 1.0)

            output = model(x_adv_tmp)
            pred_adv = torch.argmax(output, dim=1)
            if pred_adv[0] == y[0]:
                x_adv = x_adv_tmp
                break  # to next iteration
            out_gt = output[:, y[0]]
            dist = ori_logit - out_gt
            if step_size < 1e-6:
                return x_adv_tmp
            print('step_size:', step_size)
            print('dist:', ori_logit, out_gt, dist)
            if dist > threshold + 0.5:
                is_valid = False
                step_size /= 2
            elif dist > threshold:
                print('Successful attacking to threshold.')
                return x_adv_tmp
            else:
                is_valid = True
                x_adv = x_adv_tmp

    print('Not successful to threshold after attacking {} steps.'.format(perturb_steps))
    return x_adv


def gene_advs(model, dataloader, args, save_dir, threshold):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    res = []
    count = 0
    with torch.no_grad():
        for i, (image_name, img, lbl) in enumerate(dataloader):
            if i >= args.selected_num:
                break
            img_name = image_name[0]
            print("img: %d" % i, img_name, 'label:', lbl)
            B, N = img.shape[0], args.num_points
            img = img.to(device)  # (B, 1024, 3), here B=1
            lbl = lbl.to(device)  # (B,)
            img = img.permute(0, 2, 1)  # (B,3,1024)

            ori_output = model(img)  # (B,Classes=10)
            pred_ori = torch.argmax(ori_output, dim=1)
            pred_logit = ori_output[:, pred_ori[0]]
            print('pred:', pred_ori, 'pred logit:', pred_logit)

            if pred_ori.item() == lbl.item():
                count += 1
                img_adv = attack_magnitude_untarget(model, img, lbl, pred_logit, device, step_size=args.step_size, epsilon=args.epsilon,
                                           perturb_steps=args.perturb_steps, threshold=threshold, distance=args.distance).to(device)
                output = model(img_adv)
                pred_adv = torch.argmax(output, dim=1)
                out_gt = output[:, lbl[0]]
                dist = pred_logit - out_gt
                if dist > threshold and dist <= (threshold + 0.5):
                    print("Dist", dist)
                    delta = torch.abs(img_adv - img)
                    avg_delta = torch.sqrt(torch.norm(delta, p=2).pow(2) / (N * 3))
                    img_adv = img_adv.detach().cpu().numpy()
                    np.save(save_dir + '/{}.npy'.format(img_name), img_adv)
                    tmp = "{}  label:{}  untarget label:{}  avg delta:{}".format(img_name, lbl.item(),
                                                                                 pred_adv.item(),
                                                                                 avg_delta.item())
                    res.append(tmp)
            else:
                print('Predict incorrectly.')
            print('----------------------------')

    print(count, 'images are classified correctly.')
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--root', type=str, default='data/modelnet40_numpy', help='data path')
    parser.add_argument('--arch', type=str, default='pointnet2')
    parser.add_argument('--dataset', type=str, default='modelnet40')
    parser.add_argument('--model_path', default="model_best.t7", type=str, help='model path')

    parser.add_argument('--num_points', type=int, default=1024, help='num of points to use')
    parser.add_argument("--selected_num", default=10, type=int)
    parser.add_argument('--distance', default='l_inf', type=str)
    parser.add_argument('--epsilon', default=0.1, type=float)
    parser.add_argument('--step_size', default=0.001, type=float)
    parser.add_argument('--perturb_steps', default=1000, type=int)
    parser.add_argument('--threshold', default=1.0, type=float)
    parser.add_argument('--save_dir', default='advImgs_untarget', type=str, help="path for generated adv samples")

    args = parser.parse_args()
    set_random(args.seed)

    prefix = os.path.join("{}-{}".format(args.arch, args.dataset))
    save_dir = os.path.join(prefix, args.save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model, dataloader = prepare(args)
    res = gene_advs(model, dataloader, args, save_dir, args.threshold)

    with open(save_dir + '/result.txt', 'a') as f:
        for im in res:
            f.writelines(im + '\n')






