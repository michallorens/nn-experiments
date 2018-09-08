import os.path as osp
import random
import itertools
import numpy as np
from enum import Enum

from zipfile import ZipFile
from glob import glob

import copy
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.transforms.transforms import Scale
from datasets.viper import Mode


class CUHK01(Dataset):
    path = './data/campus/'
    zip_path = './data/CUHK01.zip'

    def __init__(self, split, transform):
        self.mode = Mode.TEST
        self.transform = transform
        self.samples = []

        if not osp.exists(self.path) and osp.exists(self.zip_path):
            with ZipFile(self.zip_path, 'r') as zip_file:
                zip_file.extractall('./data')

        files_a = sorted(glob(self.path + '*001.png'))
        files_b = sorted(glob(self.path + '*003.png'))

        files = sorted(zip(files_a, files_b))
        random.shuffle(files)
        files_a[:], files_b[:] = zip(*files)

        a = files_a[split[0]:split[1]]
        a = {int(path.split('\\')[1].split('.')[0][:-3]) : path for path in a}
        b = files_b[split[0]:split[1]]
        b = {int(path.split('\\')[1].split('.')[0][:-3]) : path for path in b}

        for id, cam_a in a.items():
            cam_a = [id, cam_a]
            cam_b = [id, b[id]]

            cam_n = [[b[int(n_id)], b[int(n_id)]] for n_id in b.keys() if n_id != id]

            triples = itertools.product([cam_a], [cam_b], cam_n)
            for anchor, positive, negative in triples:
                self.samples.append([anchor, positive, negative])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        sample = copy.deepcopy(self.samples[item])

        for el in sample:
            el[1:] = [Image.open(path) for path in el[1:]]

        sample[0][1] = self.transform(sample[0][1])

        sample[1][1] = self.transform(sample[1][1])
        sample[2][1] = self.transform(sample[2][1])

        return sample

    @staticmethod
    def create(split, transform=transforms.Compose([
        Scale(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])):

        test = CUHK01((split[0], None), transform)
        return None, None, test
