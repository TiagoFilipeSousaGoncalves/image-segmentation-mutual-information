# Imports
import random
import numpy as np
import math
from PIL import Image

# PyTorch Imports
import torch
from torchvision import transforms



# Class: Resize Function
class Resize:
    def __init__(self, size):
        self.t = transforms.Resize(size)

    def __call__(self, sample):
        img, seg = sample

        img = self.t(img)
        seg = self.t(seg)

        return img, seg



# Class: RandomCrop Function
class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, seg = sample

        width, height = img.size

        new_height, new_width = self.size

        left = random.randint(0, width-new_width)
        top = random.randint(0, height-new_height)

        img = img.crop((left, top, left+new_width, top+new_height))
        seg = seg.crop((left, top, left+new_width, top+new_height))

        return img, seg



# Class: CenterCrop Function
class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, seg = sample

        width, height = img.size

        new_height, new_width = self.size

        left = math.floor((width-new_width)/2)
        top = math.floor((height-new_height)/2)

        img = img.crop((left, top, left+new_width, top+new_height))
        seg = seg.crop((left, top, left+new_width, top+new_height))

        return img, seg



# Class: ToTensor Function
class ToTensor:
    def __init__(self):
        self.totensor = transforms.ToTensor()

    def __call__(self, sample):
        img, seg = sample

        img = self.totensor(img)

        seg = np.array(seg)
        seg = torch.from_numpy(seg)
        seg = seg.long()

        return img, seg



# Class: Normalize Function
class Normalize:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.norm = transforms.Normalize(mean=mean, std=std)

    def __call__(self, sample):
        img, seg = sample

        img = self.norm(img)

        return img, seg



# List: Color Palette for the results
color_palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0,
                 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128, 0, 64, 0,
                 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128, 64, 64, 0,
                 192, 64, 0, 64, 192, 0, 192, 192, 0, 64, 64, 128, 192, 64, 128, 64, 192, 128, 192, 192, 128, 0, 0,
                 64, 128, 0, 64, 0, 128, 64, 128, 128, 64, 0, 0, 192, 128, 0, 192, 0, 128, 192, 128, 128, 192, 64, 0,
                 64, 192, 0, 64, 64, 128, 64, 192, 128, 64, 64, 0, 192, 192, 0, 192, 64, 128, 192, 192, 128, 192, 0,
                 64, 64, 128, 64, 64, 0, 192, 64, 128, 192, 64, 0, 64, 192, 128, 64, 192, 0, 192, 192, 128, 192, 192,
                 64, 64, 64, 192, 64, 64, 64, 192, 64, 192, 192, 64, 64, 64, 192, 192, 64, 192, 64, 192, 192, 192, 192,
                 192, 32, 0, 0, 160, 0, 0, 32, 128, 0, 160, 128, 0, 32, 0, 128, 160, 0, 128, 32, 128, 128, 160, 128,
                 128, 96, 0, 0, 224, 0, 0, 96, 128, 0, 224, 128, 0, 96, 0, 128, 224, 0, 128, 96, 128, 128, 224, 128,
                 128, 32, 64, 0, 160, 64, 0, 32, 192, 0, 160, 192, 0, 32, 64, 128, 160, 64, 128, 32, 192, 128, 160, 192,
                 128, 96, 64, 0, 224, 64, 0, 96, 192, 0, 224, 192, 0, 96, 64, 128, 224, 64, 128, 96, 192, 128, 224, 192,
                 128, 32, 0, 64, 160, 0, 64, 32, 128, 64, 160, 128, 64, 32, 0, 192, 160, 0, 192, 32, 128, 192, 160, 128,
                 192, 96, 0, 64, 224, 0, 64, 96, 128, 64, 224, 128, 64, 96, 0, 192, 224, 0, 192, 96, 128, 192, 224, 128,
                 192, 32, 64, 64, 160, 64, 64, 32, 192, 64, 160, 192, 64, 32, 64, 192, 160, 64, 192, 32, 192, 192, 160,
                 192, 192, 96, 64, 64, 224, 64, 64, 96, 192, 64, 224, 192, 64, 96, 64, 192, 224, 64, 192, 96, 192, 192,
                 224, 192, 192, 0, 32, 0, 128, 32, 0, 0, 160, 0, 128, 160, 0, 0, 32, 128, 128, 32, 128, 0, 160, 128,
                 128, 160, 128, 64, 32, 0, 192, 32, 0, 64, 160, 0, 192, 160, 0, 64, 32, 128, 192, 32, 128, 64, 160,
                 128, 192, 160, 128, 0, 96, 0, 128, 96, 0, 0, 224, 0, 128, 224, 0, 0, 96, 128, 128, 96, 128, 0, 224,
                 128, 128, 224, 128, 64, 96, 0, 192, 96, 0, 64, 224, 0, 192, 224, 0, 64, 96, 128, 192, 96, 128, 64, 224,
                 128, 192, 224, 128, 0, 32, 64, 128, 32, 64, 0, 160, 64, 128, 160, 64, 0, 32, 192, 128, 32, 192, 0, 160,
                 192, 128, 160, 192, 64, 32, 64, 192, 32, 64, 64, 160, 64, 192, 160, 64, 64, 32, 192, 192, 32, 192, 64,
                 160, 192, 192, 160, 192, 0, 96, 64, 128, 96, 64, 0, 224, 64, 128, 224, 64, 0, 96, 192, 128, 96, 192, 0,
                 224, 192, 128, 224, 192, 64, 96, 64, 192, 96, 64, 64, 224, 64, 192, 224, 64, 64, 96, 192, 192, 96, 192,
                 64, 224, 192, 192, 224, 192, 32, 32, 0, 160, 32, 0, 32, 160, 0, 160, 160, 0, 32, 32, 128, 160, 32, 128,
                 32, 160, 128, 160, 160, 128, 96, 32, 0, 224, 32, 0, 96, 160, 0, 224, 160, 0, 96, 32, 128, 224, 32, 128,
                 96, 160, 128, 224, 160, 128, 32, 96, 0, 160, 96, 0, 32, 224, 0, 160, 224, 0, 32, 96, 128, 160, 96, 128,
                 32, 224, 128, 160, 224, 128, 96, 96, 0, 224, 96, 0, 96, 224, 0, 224, 224, 0, 96, 96, 128, 224, 96, 128,
                 96, 224, 128, 224, 224, 128, 32, 32, 64, 160, 32, 64, 32, 160, 64, 160, 160, 64, 32, 32, 192, 160, 32,
                 192, 32, 160, 192, 160, 160, 192, 96, 32, 64, 224, 32, 64, 96, 160, 64, 224, 160, 64, 96, 32, 192, 224,
                 32, 192, 96, 160, 192, 224, 160, 192, 32, 96, 64, 160, 96, 64, 32, 224, 64, 160, 224, 64, 32, 96, 192,
                 160, 96, 192, 32, 224, 192, 160, 224, 192, 96, 96, 64, 224, 96, 64, 96, 224, 64, 224, 224, 64, 96, 96,
                 192, 224, 96, 192, 96, 224, 192, 224, 224, 192]



# Function: Show masked image
def save_masked_img(img, fname):
    img = img.cpu()

    new_img = Image.fromarray(img.to(torch.uint8).numpy(), mode='P')
    new_img.putpalette(color_palette)

    new_img.save(fname)

    return



# Function: Show original image
def save_original_img(img, fname):
    img = img.cpu()

    t = transforms.ToPILImage()

    new_img = t(img)

    new_img.save(fname)

    return



# Function: Compute Metrics
def compute_metrics(dataloader, model, device, n_labels, n_epochs=5, is_unet=True):

    labels_list = []
    predicted_list = []

    model.eval()

    with torch.no_grad():
        for n in range(n_epochs):
            for i, data in enumerate(dataloader):
                images, labels = data

                images = images.to(device)
                labels = labels.to(device)

                if not is_unet:
                    predicted_labels = model(images)['out']
                
                else: 
                    predicted_labels = model(images)
                
                predicted_labels = torch.argmax(predicted_labels, dim=1)

                labels_list.append(labels.cpu())
                predicted_list.append(predicted_labels.cpu())

                # print(f'Inference with Random Crop: Epoch:{n} | Mini-Batch: {i}')

    labels_list = torch.cat(labels_list, dim=0)
    predicted_list = torch.cat(predicted_list, dim=0)

    labels_list = labels_list.view(-1)
    predicted_list = predicted_list.view(-1)

    mask = labels_list != 255

    labels_list = labels_list[mask]
    predicted_list = predicted_list[mask]


    # Create IoU variable
    iou = 0

    for l in range(n_labels):
        # print(f"IoU Computation | Current label {l}")
        pred_inds = predicted_list == l
        target_inds = labels_list == l

        intersection = pred_inds & target_inds
        union = pred_inds | target_inds

        n_intersection = torch.sum(intersection).item()
        n_union = torch.sum(union).item()

        iou += (n_intersection / n_union)
    

    # Final IoU is the mean of all the IoU per channel
    iou = iou / n_labels

    # Accuracy
    acc = labels_list == predicted_list
    acc = torch.sum(acc)/acc.size()[0]


    return iou, acc