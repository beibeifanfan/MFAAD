from torchvision import transforms
from PIL import Image
import os
import torch
import glob
import numpy as np
from datasets.noise import HairNoiseGenerator, generate_hair_curve, add_single_bspline_noise
import cv2
import matplotlib.pyplot as plt

height = 256
width = 608


class ToTensor(object):
    def __call__(self, image):
        try:
            image = torch.from_numpy(image.transpose(2, 0, 1))
        except:
            print('Invalid_transpose, please make sure images have shape (H, W, C) before transposing')
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        return image


class Normalize(object):
    """
    Only normalize images
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image):
        image = (image - self.mean) / self.std
        return image


def get_data_transforms(size, isize):
    data_transforms = transforms.Compose([Normalize(), ToTensor()])  # 这个操作将数据转换为PyTorch张量的格式

    gt_transforms = transforms.Compose([
        transforms.Resize((size, isize)),
        transforms.ToTensor()])
    return data_transforms, gt_transforms


class MVTecDataset_train(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.img_path = root
        self.transform = transform
        # load dataset
        self.img_paths = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_paths = glob.glob(os.path.join(self.img_path, 'good') + "/*.bmp")
        return img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img / 255., (width, height))  # 对图像进行了归一化（除以了255.）和缩放操作，将其调整为256x256的尺寸
        ## Normal
        img_normal = self.transform(img)
        b_img_noise = add_single_bspline_noise(img, num_hairs=1 , color=(255, 255, 255))
        img_noise = img - b_img_noise
        img_noise1 = self.transform(img_noise)

        return img_normal, img_noise1, img_path.split('/')[-1]


class MVTecDataset_test(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform):
        self.img_path = os.path.join(root, 'test')
        self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img / 255., (width, height))
        ## Normal
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.shape[-2], img.shape[-1]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)
        # print(img.shape,gt.shape)
        assert img.shape[1:] == gt.shape[1:], "image.size != gt.size !!!"

        return (img, gt, label, img_type, img_path.split('/')[-1])



