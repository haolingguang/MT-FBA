import torch.utils.data as data
import torchvision.transforms as transforms
import torch

import os
from PIL import Image
# import pandas as pd
# from torchvision.transforms.transforms import CenterCrop

IMG_EXTENSIONS = ['.png', '.jpg']

def cifar_trans():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    return tf


def find_inputs(folder, true_label, types=IMG_EXTENSIONS):
    inputs = []
    for root, _, files in os.walk(folder, topdown=False):
        for rel_filename in files:
            _, ext = os.path.splitext(rel_filename)
            if ext.lower() in types:
                abs_filename = os.path.join(root, rel_filename)
                inputs.append((abs_filename, true_label, rel_filename))
                
    return inputs


class Dataset(data.Dataset):
    
    def __init__(self, root_back, transform=cifar_trans()):
        
        # true_label = int(os.path.basename(root_back))

        # 如果是DBA直接给标签
        true_label = 1

        back_imgs = find_inputs(root_back, true_label)
        if len(back_imgs) == 0:
                raise(RuntimeError("Found 0 images in subfolders of: " + root_back + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        
        self.root_back = root_back
        self.back_imgs = back_imgs
        self.transform = transform


    def __getitem__(self, index):
        back_path, target, filename = self.back_imgs[index]
        back_img = Image.open(back_path).convert('RGB')
        
        if self.transform is not None:
            back_img = self.transform(back_img)
        if target is None:
            target = torch.zeros(1).long()
        return back_img, target, filename
    
    def __len__(self):
        return len(self.back_imgs)

    
# if __name__ =='__main__':
#     dataset=Dataset('./save/backdoor_image/9')
#     print(1)
