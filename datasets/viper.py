import os.path as osp
import random
import itertools
from enum import Enum

from zipfile import ZipFile
from glob import glob

import copy
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.transforms.transforms import Scale


class Mode(Enum):
    TRAIN = 0,
    VALIDATE = 1,
    TEST = 2


class VIPeR(Dataset):
    path = './data/VIPeR/'
    zip_path = './data/VIPeR.v1.0.zip'

    def __init__(self, mode, split, transform, train_transform, negative_samples=3, shuffle_seed=None):
        self.mode = mode
        self.transform = transform
        self.train_transform = train_transform
        self.samples = []

        if not osp.exists(self.path) and osp.exists(self.zip_path):
            with ZipFile(self.zip_path, 'r') as zip_file:
                zip_file.extractall('./data')

        files_a = sorted(glob(self.path + 'cam_a/*.bmp'))
        files_b = sorted(glob(self.path + 'cam_b/*.bmp'))

        files = list(zip(files_a, files_b))
        if shuffle_seed is None:
            random.shuffle(files)
        else:
            random.Random(shuffle_seed).shuffle(files)
        files_a[:], files_b[:] = zip(*files)

        a = files_a[split[0]:split[1]]
        a = {path.split('\\')[1].split('_')[0] : path for path in a}
        b = files_b[split[0]:split[1]]
        b = {path.split('\\')[1].split('_')[0] : path for path in b}

        for id, cam_a in a.items():
            cam_a = [id, cam_a]
            cam_b = [id, b[id]]

            if self.mode == Mode.TRAIN:
                if negative_samples is not None:
                    n_ids = random.sample(a.keys(), negative_samples)
                    while id in n_ids:
                        n_ids = random.sample(a.keys(), negative_samples)
                else:
                    n_ids = [n for n in a.keys() if n != id]

                cam_n = [[n_id, b[n_id]] for n_id in n_ids]
            else:
                cam_n = [[n_id, b[n_id]] for n_id in a.keys() if n_id != id]

            triples = itertools.product([cam_a], [cam_b], cam_n)
            # if self.mode == Mode.VALIDATE:
            #     triples = random.sample(list(triples), 10)
            for anchor, positive, negative in triples:
                self.samples.append([anchor, positive, negative])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        sample = copy.deepcopy(self.samples[item])

        for el in sample:
            el[1:] = [Image.open(path) for path in el[1:]]

        if self.mode == Mode.TRAIN:
            sample[0][1] = self.train_transform(sample[0][1])
        else:
            sample[0][1] = self.transform(sample[0][1])

        sample[1][1] = self.transform(sample[1][1])
        sample[2][1] = self.transform(sample[2][1])

        return sample

    @staticmethod
    def create(split, negative_samples=None, shuffle_seed=None, transform=transforms.Compose([
                    Scale(64),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ]),
               train_transform=transforms.Compose([
                    transforms.RandomSizedCrop(64),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])):

        train = VIPeR(Mode.TRAIN, (0, split[0]), transform, train_transform, negative_samples=negative_samples, shuffle_seed=shuffle_seed)
        val = VIPeR(Mode.VALIDATE, split, transform, train_transform, shuffle_seed=shuffle_seed)
        test = VIPeR(Mode.TEST, (split[0], None), transform, train_transform, shuffle_seed=shuffle_seed)
        return train, val, test
