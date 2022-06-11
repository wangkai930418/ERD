import os.path as osp

# +
import PIL
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import ImageEnhance

transformtypedict = dict(
    Brightness=ImageEnhance.Brightness,
    Contrast=ImageEnhance.Contrast,
    Sharpness=ImageEnhance.Sharpness,
    Color=ImageEnhance.Color,
)


class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [
            (transformtypedict[k], transformdict[k]) for k in transformdict
        ]

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (randtensor[i] * 2.0 - 1.0) + 1
            out = transformer(out).enhance(r).convert("RGB")

        return out



THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, ".."))
IMAGE_PATH = osp.join(ROOT_PATH, "data/cub/images")
SPLIT_PATH = osp.join(ROOT_PATH, "data/cub/split")

class CIFAR100(Dataset):
    def __init__(self, setname, args, data, label,location=163):
        self.data = data
        self.label = label
        self.num_class = len(set(label))
        self.setname=setname

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        if setname == "train":
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(84),
                    ImageJitter(dict(Brightness=0.4, Contrast=0.4, Color=0.4)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
                    ),
                ]
            )

        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(84, interpolation=PIL.Image.BICUBIC),
                    transforms.CenterCrop(84),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        img = Image.fromarray(path)
        image = self.transform(img)
        return image, label


class Hard_Mine_CIFAR(Dataset):
    def __init__(self, setname, args, data, label,location=122):
        self.data = data
        self.label = label
        self.num_class = len(set(label))
        self.location=location
        # Transformation
        self.setname=setname
        image_size = 84

        if setname == "train":
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(image_size),
                    ImageJitter(dict(Brightness=0.4, Contrast=0.4, Color=0.4)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(92),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
                    ),
                ]
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        img = Image.fromarray(path)
        image = self.transform(img)
        return image, label, path