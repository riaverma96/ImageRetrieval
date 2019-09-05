from __future__ import print_function
import os
import re
import json
import cPickle
import numpy as np
import h5py
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from PIL import Image
from skimage import io
import torchvision.transforms as transforms
import pdb
from os import path


base_path = "../data/"
image_feature_file = "../data/img_features.pkl"
train_image_feature_file = "../data/img_features_train.pkl"
val_image_feature_file = "../data/img_features_val.pkl"

def process_attributes(base_path=None, old_filename=None, new_filename=None):
    """
    Processes attribute file to be CSV compatible. Saves in new file in the same directory.
    """
    attr = pd.read_csv(base_path + 'Anno/list_attr_items.txt', skiprows=2, sep='\s+', names=['image_name'] + ['attr_%d' % i for i in range(1000)])
    attr.replace(-1, 0, inplace=True)
    attr.to_csv(base_path + 'Anno/list_attr_img_without_commas.txt', index=False)


def get_image_feature(model_conv, img_filepath):
    img = Image.open(img_filepath)
    imsize = 224
    loader = transforms.Compose([transforms.Resize((imsize, imsize)),
                                 transforms.ToTensor()])
    img = loader(img)
    img = img.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
    img = Variable(img)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        img = img.cuda()  # assumes we're using GPU
        model_conv = model_conv.cuda()
    img_feature = model_conv(img)
    return img_feature


def _load_attributes(name, data_splits):
    id_attributes = {}
    # let's only train on 8 images
    img_cnt = 0
    with open(base_path + 'Anno/list_attr_items.txt') as f:
        next(f)  # skip the header
        next(f)
        for line in f:
            if img_cnt > 7:
                break
            img_id = int(line.split(' ')[0].split('_')[1])
            if data_splits[img_id] == name:
                attributes = line.split(',')[1:]
                id_attributes[img_id] = attributes
                img_cnt += 1
    return id_attributes


def _load_and_extract_image_features(model_conv, name, data_splits):
    img_features = {}
    img_cnt = 0
    for subdir, dirs, files in os.walk(base_path + 'img/WOMEN/Dresses/'):
        for file in files:
            if img_cnt > 20:
                break
            if not file.startswith('.'):
                # only use anchor images (i.e. 'front' images)
                # img_type = file.split('_')[2].split('.')[0]
                # if img_type == 'front':
                img_id = subdir.split('/')[-1].split('_')[1]
                if data_splits[img_id] == name:
                    print(data_splits[img_id])
                    img_features[int(img_id)] = get_image_feature(model_conv, os.path.join(subdir, file))
                    img_cnt += 1
    cPickle.dump(img_features, open(image_feature_file, 'wb'))
    return img_features


def _process_train_val_splits():
    splits = {}
    with open(base_path + 'Eval/list_eval_partition.txt') as f:
        next(f)  # skip the header
        next(f)
        for line in f:
            if line.find('train') != -1:
                split = 'train'
            elif line.find('gallery') != -1:
                split = 'val'
            img_id = int(line.split(' ')[0].split('/')[-2].split('_')[1])
            splits[img_id] = split
    return splits


# TODO: Split data into train/val/test.
class ImageRetrievalDataset(Dataset):
    def __init__(self, name, model_conv):
        super(ImageRetrievalDataset, self).__init__()

        assert name in ['train', 'val']
        self.data_splits = _process_train_val_splits()
        self.attributes = _load_attributes(name, self.data_splits)
        # image_feature_file = train_image_feature_file if name == 'train' else val_image_feature_file
        if path.exists(image_feature_file):
            self.img_features = cPickle.load(open(image_feature_file))
        else:
            self.img_features = _load_and_extract_image_features(model_conv, name, self.data_splits)

        self.num_ans_candidates = 463
        self.v_dim = 4096  # switch from hard-coded. img_feature is [1x4096] dim.

    def __getitem__(self, index):
        category = None  # torch.tensor(category)
        img_features = self.img_features[index]
        attributes = self.attributes[index]
        # TODO: For category prediction task, generate one-hot vectors for target classes
        # target = torch.zeros(self.num_ans_candidates)
        # if labels is not None:
        #     target.scatter_(0, labels, scores)
        return category, img_features.data, attributes, target

    def __len__(self):
        return len(self.attributes)
