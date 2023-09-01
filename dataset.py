import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from imageio import imread
from os import walk


class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_list):
        super(Dataset, self).__init__()
        self.transform_rgb = transforms.Compose([
            transforms.Resize((256, 256), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.image_list = image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        id, label, source_path, mask_path = self.image_list[index]
        # load image
        source = Image.open(source_path)
        # load mask
        mask = torch.from_numpy(
            np.array(Image.open(mask_path).resize((256, 256)).convert("1")) / 1).long()
        mask = mask.unsqueeze(0)
        return {"source": self.transform_rgb(source),
                "mask": mask,
                "label": label,
                "id": id
                }


def get_data_loader():
    image_list = []
    for i in range(1, 438):
        image_list.append(
            (0, 'benign', 'Dataset_BUSI_with_GT/benign/benign ({}).png'.format(i),
             'Dataset_BUSI_with_GT/benign/benign ({})_mask.png'.format(i)))
    for i in range(1, 210):
        image_list.append(
            (1, 'malignant', 'Dataset_BUSI_with_GT/malignant/malignant ({}).png'.format(i),
             'Dataset_BUSI_with_GT/malignant/malignant ({})_mask.png'.format(i)))
    for i in range(1, 133):
        image_list.append(
            (2, 'normal', 'Dataset_BUSI_with_GT/normal/normal ({}).png'.format(i),
             'Dataset_BUSI_with_GT/normal/normal ({})_mask.png'.format(i)))
    dataset = Dataset(image_list)
    return DataLoader(dataset, batch_size=12, shuffle=True)
