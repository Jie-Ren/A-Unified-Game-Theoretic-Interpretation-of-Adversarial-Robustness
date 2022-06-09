import os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
import random


def set_random(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_dataset_modelnet40_10k(opt, mode):
    dataset = []

    DATA_DIR = opt.root

    if opt.dataset == 'modelnet10':
        classes = 10
    else:
        classes = 40

    f = open(os.path.join(DATA_DIR, 'modelnet%d_shape_names.txt' % classes))
    shape_list = [str.rstrip() for str in f.readlines()]
    f.close()

    if 'train' == mode:
        f = open(os.path.join(DATA_DIR, 'modelnet%d_train_dgcnn_0216.txt' % classes),
                 'r')  # rqh, revise, original: modelnet%d_train.txt
        lines = [str.rstrip() for str in f.readlines()]
        f.close()
    elif 'test' == mode:
        f = open(os.path.join(DATA_DIR, 'modelnet%d_test.txt' % classes), 'r')
        lines = [str.rstrip() for str in f.readlines()]
        f.close()
    else:
        raise Exception('Network mode error.')

    for i, name in enumerate(lines):
        # locate the folder name
        folder = name[0:-5]
        file_name = name

        # get the label
        label = shape_list.index(folder)

        item = (file_name, os.path.join(DATA_DIR, folder, file_name + '.npy'), label)
        dataset.append(item)

    return dataset


def get_folder_name_list():
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join('data', 'modelnet40_numpy')
    f = open(os.path.join(DATA_DIR, 'modelnet10_train_dgcnn_0216.txt'), 'r')
    lines = [str.rstrip() for str in f.readlines()]
    f.close()
    return lines


def square_distance(src, dst):
    """ Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


class ModelNet_Loader(Dataset):
    def __init__(self, opt, num_points, partition='train'):
        super(ModelNet_Loader, self).__init__()

        self.opt = opt
        self.partition = partition
        self.num_points = num_points

        self.dataset = make_dataset_modelnet40_10k(opt, self.partition)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        file_name, pc_np_file, class_id = self.dataset[index]

        data = np.load(pc_np_file)

        pointcloud = data[0:self.num_points, 0:3]  # Nx3

        # convert to tensor
        pointcloud = pointcloud.astype(np.float32)  # Nx3

        return file_name, pointcloud, class_id

