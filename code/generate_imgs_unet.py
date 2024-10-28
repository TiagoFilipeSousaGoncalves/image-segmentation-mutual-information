# Imports
import os
import numpy as np
import time

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# Project Imports
from datasets import VOCDataset, CityScapesDataset, ADE20k
from models import UNet
from utils import Resize, RandomCrop, ToTensor, CenterCrop, Normalize, save_original_img, save_masked_img



# Select device (GPU or CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Datasets
datasets = ['cityscapes', 'VOC', 'ADE20k']



# Loop through datasets
for dataset in datasets:
    print(f"Current Dataset: {dataset}")

    # Cityscapes Dataset
    if dataset == 'cityscapes':

        test_dataset = CityScapesDataset(
            img_root='data/CITYSCAPES/leftImg8bit',
            seg_root='data/CITYSCAPES/gtFine',
            mode='val',
            extensions=['.png'],
            transform=transforms.Compose([Resize(234), RandomCrop((224, 224)), ToTensor()])
        )
    

    # PASCALVOC2012
    elif dataset == 'VOC':

        test_dataset = VOCDataset(
            img_root='data/PASCALVOC2012/VOCdevkit/VOC2012/JPEGImages',
            seg_root='data/PASCALVOC2012/VOCdevkit/VOC2012/SegmentationClass',
            filenames_path='data/PASCALVOC2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt',
            extensions=['.jpg', '.png'],
            transform=transforms.Compose([Resize(234), RandomCrop((224, 224)), ToTensor()])
        )
    

    # ADE20K
    elif dataset == 'ADE20k':

        test_dataset = ADE20k(
            img_root='data/ADE20K/ADEChallengeData2016/images',
            seg_root='data/ADE20K/ADEChallengeData2016/annotations',
            mode='val',
            extensions=['.jpg', '.png'],
            transform=transforms.Compose([Resize(234), RandomCrop((224, 224)),ToTensor()])
        )
    


    # Set batch size
    batch_size = 1

    # Create dataloader
    dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model
    model = UNet(in_channels=3, out_channels=test_dataset.num_classes)
    model.load_state_dict(torch.load(f'results/baseline/weights/unet_model_{dataset.lower()}.pth', map_location='cpu'))
    model.to(device)


    # Put model in evaluation mode
    model.eval()


    # Create path to save figures
    figure_path = os.path.join('results/baseline/', 'figures', dataset.lower())
    if os.path.isdir(figure_path)==False:
        os.makedirs(figure_path)


    # Turn off gradient computation
    with torch.no_grad():

        img_count = 0

        for i, data in enumerate(dataloader):
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)

            predicted_labels = model(images)
            predicted_labels = torch.argmax(predicted_labels, dim=1)

            save_original_img(images[0], os.path.join(figure_path, f'{i}_img_original.png'))
            save_masked_img(labels[0], os.path.join(figure_path, f'{i}_mask_original.png'))
            save_masked_img(predicted_labels[0], os.path.join(figure_path, f'{i}_mask_predicted.png'))
        
            img_count += 1

            if img_count > 8:
                break


print("Finished.")