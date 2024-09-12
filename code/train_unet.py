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
from utils import Resize, RandomCrop, ToTensor, CenterCrop, Normalize



# Select device (GPU or CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Datasets
datasets = ['cityscapes', 'VOC', 'ADE20k']



# Loop through datasets
for dataset in datasets:
    print(f"Current Dataset: {dataset}")

    # Cityscapes Dataset
    if dataset == 'cityscapes':

        train_dataset = CityScapesDataset(
            img_root='/ctm-hdd-pool01/tgoncalv/image-segmentation-mutual-information/data/CITYSCAPES/leftImg8bit',
            seg_root='/ctm-hdd-pool01/tgoncalv/image-segmentation-mutual-information/data/CITYSCAPES/gtFine',
            mode='train',
            extensions=['.png'],
            transform=transforms.Compose([Resize(234), RandomCrop((224, 224)), ToTensor()])
        )
    

    # PASCALVOC2012
    elif dataset == 'VOC':

        train_dataset = VOCDataset(
            img_root='/ctm-hdd-pool01/tgoncalv/image-segmentation-mutual-information/data/PASCALVOC2012/VOCdevkit/VOC2012/JPEGImages',
            seg_root='/ctm-hdd-pool01/tgoncalv/image-segmentation-mutual-information/data/PASCALVOC2012/VOCdevkit/VOC2012/SegmentationClass',
            filenames_path='/ctm-hdd-pool01/tgoncalv/image-segmentation-mutual-information/data/PASCALVOC2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt',
            extensions=['.jpg', '.png'],
            transform=transforms.Compose([Resize(234), RandomCrop((224, 224)), ToTensor()])
        )
    

    # ADE20K
    elif dataset == 'ADE20k':

        train_dataset = ADE20k(
            img_root='/ctm-hdd-pool01/tgoncalv/image-segmentation-mutual-information/data/ADE20K/ADEChallengeData2016/images',
            seg_root='/ctm-hdd-pool01/tgoncalv/image-segmentation-mutual-information/data/ADE20K/ADEChallengeData2016/annotations',
            mode='train',
            extensions=['.jpg', '.png'],
            transform=transforms.Compose([Resize(234), RandomCrop((224, 224)),ToTensor()])
        )
    


    # Define train hyperparameters
    # Epochs
    epochs = 200

    # Batch size
    batch_size = 8 
    
    # Dataloader
    dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Model
    model = UNet(in_channels=3, out_channels=train_dataset.num_classes)
    # model.load_state_dict(torch.load('{model_path}', map_location='cpu'))
    model.to(device)

    # Loss function
    loss = nn.CrossEntropyLoss(ignore_index=255)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters())


    # Put model in train mode
    model.train()

    
    # Start training loop
    min_loss = np.inf

    for epoch in range(epochs):
        running_loss = 0.0
        start = time.time()

        for _, data in enumerate(dataloader):
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)

            predicted_labels = model(images)

            model_loss = loss(predicted_labels, labels)

            optimizer.zero_grad()
            model_loss.backward()
            optimizer.step()

            # Keep running loss updated
            running_loss += model_loss.item()

        
        # Compute avg running loss
        avg_running_loss = running_loss / len(dataloader.dataset)

        # Print statistics
        print(f'Epoch: {epoch + 1}/{epochs} | Loss: {avg_running_loss}')
        print(f'Elapsed Time: {time.time() - start}')


        # Save model based on loss
        if avg_running_loss < min_loss:
            
            # Weights directory
            weights_dir = "results/baseline/weights"
            if os.path.isdir(weights_dir) == False:
                os.makedirs(weights_dir)

            # Save model
            torch.save(model.state_dict(), os.path.join(weights_dir, f'unet_model_{dataset.lower()}.pth'))
            
            # Updated min loss
            min_loss = avg_running_loss