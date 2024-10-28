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
from models import UNet, DeepLabV3
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
            img_root='data/CITYSCAPES/leftImg8bit',
            seg_root='data/CITYSCAPES/gtFine',
            mode='train',
            extensions=['.png'],
            transform=transforms.Compose([Resize(234), RandomCrop((224, 224)), ToTensor()])
        )
    

    # PASCALVOC2012
    elif dataset == 'VOC':

        train_dataset = VOCDataset(
            img_root='data/PASCALVOC2012/VOCdevkit/VOC2012/JPEGImages',
            seg_root='data/PASCALVOC2012/VOCdevkit/VOC2012/SegmentationClass',
            filenames_path='data/PASCALVOC2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt',
            extensions=['.jpg', '.png'],
            transform=transforms.Compose([Resize(234), RandomCrop((224, 224)), ToTensor()])
        )
    

    # ADE20K
    elif dataset == 'ADE20k':

        train_dataset = ADE20k(
            img_root='data/ADE20K/ADEChallengeData2016/images',
            seg_root='data/ADE20K/ADEChallengeData2016/annotations',
            mode='train',
            extensions=['.jpg', '.png'],
            transform=transforms.Compose([Resize(234), RandomCrop((224, 224)),ToTensor()])
        )
    


    # Define train hyperparameters
    # Epochs
    epochs = 200

    # Batch size
    batch_size = 10
    
    # Dataloader
    dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


    # Models
    # Teacher Model
    teacher_model = DeepLabV3(out_channels=train_dataset.num_classes)
    # teacher_model.load_state_dict(torch.load('{model_path}', map_location='cpu'))
    teacher_model.to(device)

    # Student Model
    student_model = UNet(in_channels=3, out_channels=train_dataset.num_classes)
    # student_model.load_state_dict(torch.load('{model_path}', map_location='cpu'))
    student_model.to(device)


    # Loss functions
    # Teacher
    ce_loss_teacher = nn.CrossEntropyLoss(ignore_index=255)
    kldiv_loss_teacher = nn.KLDivLoss()

    # Student
    ce_loss_student = nn.CrossEntropyLoss(ignore_index=255)
    kldiv_loss_student = nn.KLDivLoss()


    # Optimizers
    # Teacher
    optimizer_teacher = torch.optim.Adam(teacher_model.parameters(), lr=1e-4)

    # Student
    optimizer_student = torch.optim.Adam(student_model.parameters(), lr=1e-4)


    # Put models in train mode
    teacher_model.train()
    student_model.train()

    
    # Start training loop
    min_loss_teacher = np.inf
    min_loss_student = np.inf

    for epoch in range(epochs):
        running_loss_teacher = 0.0
        running_loss_student = 0.0
        start = time.time()

        for _, data in enumerate(dataloader):
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)

            # Get logits
            teacher_predictions = teacher_model(images)
            student_predictions = student_model(images)

            
            # Compute Teacher Loss
            # Note: KLDiv(p_student||p_teacher)
            teacher_loss = ce_loss_teacher(teacher_predictions, labels) + kldiv_loss_teacher(torch.nn.LogSoftmax(dim=1)(teacher_model(images)), student_predictions.detach())

            # Apply backpropagation 
            optimizer_teacher.zero_grad()
            teacher_loss.backward()
            optimizer_teacher.step()



            # Get logits again
            teacher_predictions = teacher_model(images)
            student_predictions = student_model(images)


            # Compute Student Loss
            # Note: KLDiv(p_teacher||p_student)
            student_loss = ce_loss_student(student_predictions, labels) + kldiv_loss_student(torch.nn.LogSoftmax(dim=1)(student_model(images)), teacher_predictions.detach())

            # Apply backpropagation
            optimizer_student.zero_grad()
            student_loss.backward()
            optimizer_student.step()

            # Keep running loss updated
            running_loss_teacher += teacher_loss.item()
            running_loss_student += student_loss.item()
        

        # Compute avg running loss
        avg_running_loss_teacher = running_loss_teacher / len(dataloader.dataset)
        avg_running_loss_student = running_loss_student / len(dataloader.dataset)

        # Print statistics
        print(f'Epoch: {epoch + 1}/{epochs} | Teacher Loss: {avg_running_loss_teacher} | Student Loss: {avg_running_loss_student}')
        print(f'Elapsed Time: {time.time() - start}')


        # Save teacher model based on loss
        if avg_running_loss_teacher < min_loss_teacher:
            
            # Weights directory
            weights_dir = "results/mutual_learning/weights"
            if os.path.isdir(weights_dir) == False:
                os.makedirs(weights_dir)

            # Save model
            torch.save(teacher_model.state_dict(), os.path.join(weights_dir, f'teacher_deeplv3_model_{dataset.lower()}.pt'))
            
            # Updated min loss
            min_loss_teacher = avg_running_loss_teacher
        


        # Save student model based on loss
        if avg_running_loss_student < min_loss_student:
            
            # Weights directory
            weights_dir = "results/mutual_learning/weights"
            if os.path.isdir(weights_dir) == False:
                os.makedirs(weights_dir)

            # Save model
            torch.save(student_model.state_dict(), os.path.join(weights_dir, f'student_unet_model_{dataset.lower()}.pt'))

            # Updated min loss
            min_loss_student = avg_running_loss_student