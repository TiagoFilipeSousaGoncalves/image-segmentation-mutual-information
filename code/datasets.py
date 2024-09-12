# Imports
import os
from PIL import Image
import numpy as np

# PyTorch Imports
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Project Imports
from utils import Resize, RandomCrop, ToTensor, Normalize, save_masked_img, save_original_img



# Class: VOC Dataset
class VOCDataset(Dataset):

    def __init__(self, img_root, seg_root, filenames_path, extensions, transform):
        filenames = []

        with open(filenames_path, "r") as f:
            for line in f:
                line = line.replace('\n', '')
                filenames.append(line)

        img_names = []

        for image in os.listdir(img_root):
            if os.path.splitext(image)[0] in filenames and os.path.splitext(image)[1] in extensions:
                img_names.append(image)

        seg_names = []

        for image in os.listdir(seg_root):
            if os.path.splitext(image)[0] in filenames and os.path.splitext(image)[1] in extensions:
                seg_names.append(image)

        img_names.sort()
        seg_names.sort()

        self.files = []

        assert len(img_names) == len(seg_names), 'images and segmentations are not the same length'

        for img, seg in zip(img_names, seg_names):
            assert os.path.splitext(img)[0] == os.path.splitext(seg)[0], 'image and segmentation filenames do not match'
            img = os.path.join(img_root, img)
            seg = os.path.join(seg_root, seg)
            self.files.append([img, seg])

        self.transform = transform
        self.num_classes = 21

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path, seg_path = self.files[idx]

        img = Image.open(img_path)
        seg = Image.open(seg_path)

        img, seg = self.transform((img, seg))

        return img, seg



# Class: CityScapes Dataset
class CityScapesDataset(Dataset):

    # mode should be 'train', 'val' or 'test'
    def __init__(self, img_root, seg_root, mode, extensions, transform):
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
        '''Label = namedtuple('Label', [

            'name',  # The identifier of this label, e.g. 'car', 'person', ... .
            # We use them to uniquely name a class

            'id',  # An integer ID that is associated with this label.
            # The IDs are used to represent the label in ground truth images
            # An ID of -1 means that this label does not have an ID and thus
            # is ignored when creating ground truth images (e.g. license plate).
            # Do not modify these IDs, since exactly these IDs are expected by the
            # evaluation server.

            'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
            # ground truth images with train IDs, using the tools provided in the
            # 'preparation' folder. However, make sure to validate or submit results
            # to our evaluation server using the regular IDs above!
            # For trainIds, multiple labels might have the same ID. Then, these labels
            # are mapped to the same class in the ground truth images. For the inverse
            # mapping, we use the label that is defined first in the list below.
            # For example, mapping all void-type classes to the same ID in training,
            # might make sense for some approaches.
            # Max value is 255!

            'category',  # The name of the category that this label belongs to

            'categoryId',  # The ID of this category. Used to create ground truth images
            # on category level.

            'hasInstances',  # Whether this label distinguishes between single instances or not

            'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
            # during evaluations or not

            'color',  # The color of this label
        ])'''

        # Please adapt the train IDs as appropriate for your approach.
        # Note that you might want to ignore labels with ID 255 during training.
        # Further note that the current train IDs are only a suggestion. You can use whatever you like.
        # Make sure to provide your results using the original IDs and not the training IDs.
        # Note that many IDs are ignored in evaluation and thus you never need to predict these!

        labels = [
            #       name:0             id:1   trainId:2   category:3        catId:4   hasInstances:5  ignoreInEval:6    color:7
            [  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ],
            [  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ],
            [  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ],
            [  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ],
            [  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ],
            [  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ],
            [  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ],
            [  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ],
            [  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ],
            [  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ],
            [  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ],
            [  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ],
            [  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ],
            [  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ],
            [  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ],
            [  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ],
            [  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ],
            [  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ],
            [  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ],
            [  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ],
            [  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ],
            [  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ],
            [  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ],
            [  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ],
            [  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ],
            [  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ],
            [  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ],
            [  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ],
            [  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ],
            [  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ],
            [  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ],
            [  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ],
            [  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ],
            [  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ],
            [  'license plate'        , -1 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ]
        ]

        self.id2label = {label[1]: label for label in labels}
        self.trainId2label = {label[2]: label for label in reversed(labels)}

        assert mode == 'train' or mode == 'val' or mode == 'test', 'incorrect mode selected'

        img_root = os.path.join(img_root, mode)
        seg_root = os.path.join(seg_root, mode)

        img_paths = []

        for city in os.listdir(img_root):
            city_path = os.path.join(img_root, city)

            for img in os.listdir(city_path):

                if os.path.splitext(img)[1] in extensions:
                    img_path = os.path.join(city_path, img)
                    img_paths.append(img_path)

        seg_paths = []

        for city in os.listdir(seg_root):
            city_path = os.path.join(seg_root, city)

            for img in os.listdir(city_path):

                if os.path.splitext(img)[1] in extensions and 'label' in img:
                    seg_path = os.path.join(city_path, img)
                    seg_paths.append(seg_path)

        img_paths.sort(key=lambda x: os.path.split(x)[1])
        seg_paths.sort(key=lambda x: os.path.split(x)[1])

        self.files = []

        assert len(img_paths) == len(seg_paths), 'images and segmentations are not the same length'

        for img, seg in zip(img_paths, seg_paths):
            self.files.append([img, seg])

        self.transform = transform
        self.num_classes = 19

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path, seg_path = self.files[idx]

        img = Image.open(img_path)

        seg = Image.open(seg_path)
        seg = np.array(seg)

        for key in self.id2label.keys():
            seg[seg == key] = self.id2label[key][2]

        seg = Image.fromarray(seg, mode='P')

        img, seg = self.transform((img, seg))

        return img, seg



# Class: ADE20K Dataset
class ADE20k(Dataset):

    # mode should be 'train', 'val' or 'test'
    def __init__(self, img_root, seg_root, mode, extensions, transform):

        assert mode == 'train' or mode == 'val', 'incorrect mode selected'

        modes = {'train': 'training',
                 'val': 'validation'}

        mode = modes[mode]

        img_root = os.path.join(img_root, mode)
        seg_root = os.path.join(seg_root, mode)

        img_paths = []

        for img in os.listdir(img_root):

            if os.path.splitext(img)[1] in extensions:
                img_path = os.path.join(img_root, img)
                img_paths.append(img_path)

        seg_paths = []

        for img in os.listdir(seg_root):

            if os.path.splitext(img)[1] in extensions:
                seg_path = os.path.join(seg_root, img)
                seg_paths.append(seg_path)

        img_paths.sort(key=lambda x: os.path.split(x)[1])
        seg_paths.sort(key=lambda x: os.path.split(x)[1])

        self.files = []

        assert len(img_paths) == len(seg_paths), 'images and segmentations are not the same length'

        for img, seg in zip(img_paths, seg_paths):
            self.files.append([img, seg])

        self.transform = transform
        self.num_classes = 150

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path, seg_path = self.files[idx]

        img = Image.open(img_path)
        img = img.convert('RGB')

        seg = Image.open(seg_path)
        seg = np.array(seg)

        seg -= 1

        seg = Image.fromarray(seg, mode='P')

        img, seg = self.transform((img, seg))

        return img, seg