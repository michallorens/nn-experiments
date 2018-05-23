import os.path as osp
import random
import itertools
from enum import Enum

from zipfile import ZipFile
from glob import glob

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.transforms.transforms import Scale


class Mode(Enum):
    TRAIN = 0,
    VALIDATE = 1,


class CUHK01(Dataset):
    path = './data/campus/'
    zip_path = './data/CUHK01.zip'

    def __init__(self, mode, split, test_transform, train_transform, negative_samples=3):
        self.mode = mode
        self.test_transform = test_transform
        self.train_transform = train_transform
        self.samples = []

        if not osp.exists(self.path) and osp.exists(self.zip_path):
            with ZipFile(self.zip_path, 'r') as zip_file:
                zip_file.extractall('./data')

        identities = sorted(glob(self.path + '*001.png'))[split[0]:split[1]]
        identities = [id.split('\\')[1].replace('001.png', '') for id in identities]

        for id in identities:
            cam_a = ['{}{:0>4d}{:0>3d}.png'.format(self.path, int(id), i) for i in [1, 2]]
            cam_b = ['{}{:0>4d}{:0>3d}.png'.format(self.path, int(id), i) for i in [3, 4]]

            if self.mode == Mode.TRAIN:
                n_ids = random.sample(identities, negative_samples)
                while id in n_ids:
                    n_ids = random.sample(identities, negative_samples)
                cam_n = ['{}{:0>4d}{:0>3d}.png'.format(self.path, int(n_id), random.randint(1, 4)) for n_id in n_ids]
            else:
                n_ids = [n_id for n_id in identities if n_id != id]
                n_ids = itertools.product(n_ids, [1, 2, 3, 4])
                cam_n = ['{}{:0>4d}{:0>3d}.png'.format(self.path, int(i), j) for i, j in n_ids]

            triples = itertools.product(cam_a, cam_b, cam_n)
            if self.mode == Mode.VALIDATE:
                triples = random.sample(list(triples), negative_samples)
            for anchor, positive, negative in triples:
                self.samples.append([anchor, positive, negative])

            # [self.samples.append([s, id]) for s in cam_a + cam_b]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        sample = self.samples[item]
        sample = [Image.open(path) for path in sample]

        if self.mode == Mode.TRAIN:
            sample[0] = self.train_transform(sample[0])
        else:
            sample[0] = self.test_transform(sample[0])

        sample[1] = self.test_transform(sample[1])
        sample[2] = self.test_transform(sample[2])

        return sample

    @staticmethod
    def create(split, test_transform=transforms.Compose([
                    Scale(64),
                    transforms.RandomHorizontalFlip(),
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

        train = CUHK01(Mode.TRAIN, (0, split[0]), test_transform, train_transform, negative_samples=1)
        val = CUHK01(Mode.VALIDATE, split, test_transform, train_transform, negative_samples=10)
        test = CUHK01(Mode.VALIDATE, (split[0], None), test_transform, train_transform)
        return train, val, test
