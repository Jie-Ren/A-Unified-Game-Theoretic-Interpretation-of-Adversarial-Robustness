import os
import random
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def normalize(x):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).expand(x.shape[0], 3, x.shape[2], x.shape[2]).to(x.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).expand(x.shape[0], 3, x.shape[2], x.shape[2]).to(x.device)
    out = (x - mean) / std
    return out

def update_lr(optimizer, lr):
    for ix, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr[0]
    return


class ImageNetSelectedDataset(Dataset):
    def __init__(self, train, args, selected_num=60):
        assert not train, 'ImageNet train dataset does not exist'
        self.root = os.path.join(args.root, 'val')
        self.img_class = os.listdir(self.root)
        self.img_class.sort()
        self.imgs = []
        self.labels = []
        self.transformation = args.transform

        seed_torch(args.seed)
        for i in range(len(self.img_class)):
            cls = self.img_class[i]
            imgs = os.listdir(os.path.join(self.root, cls))
            imgs.sort()
            self.imgs.extend(imgs)
            labels = [i for k in range(len(imgs))]
            self.labels.extend(labels)

        img_id = random.sample(list(range(len(self.imgs))), selected_num)
        self.name_list, self.label_list = [], []

        for i in img_id:
            self.name_list.append(self.imgs[i]), self.label_list.append(self.labels[i])

        self.size = len(self.name_list)

    def __getitem__(self, idx):
        label = self.label_list[idx]
        image = self.transformation(
            Image.open(self.root + '/{}/'.format(self.img_class[label]) + self.name_list[idx]).convert('RGB'))
        image_name = self.name_list[idx]
        return image_name, image, label

    def __len__(self):
        return self.size
