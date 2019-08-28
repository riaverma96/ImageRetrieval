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


base_path = "./data/"


def process_attributes(base_path=None, old_filename=None, new_filename=None):
    """
    Processes attribute file to be CSV compatible. Saves in new file in the same directory.
    """
    attr = pd.read_csv(base_path + 'list_attr_img.txt', skiprows=2, sep='\s+', names=['image_name'] + ['attr_%d' % i for i in range(1000)])
    attr.replace(-1, 0, inplace=True)
    attr.to_csv(base_path + 'attr_info.csv', index=False)


def get_img_fields(img_filepath):
    """
    Gets image category and image id from image file path. Expects file path to
    be in following format:  img/<category>/img_<img_id>.jpg
    """
    dir = img_filepath.split('/')
    category = dir[1]
    image_id = dir[-1].split('.')[0].split('_')[1]
    return category, image_id


def get_image_feature(model_conv, img_filepath):
    img = Image.open(base_path + img_filepath)
    imsize = 224
    loader = transforms.Compose([transforms.Resize((imsize, imsize)),
                                 transforms.ToTensor()])
    img = loader(img)
    img = img.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
    img = Variable(img)

    # img = img.cuda()  # assumes we're using GPU
    img_feature = model_conv(img)
    return img_feature


# TODO: Complete
def _create_entry(model_conv, img_filepath, attributes):
    category, image_id = get_img_fields(img_filepath)
    img_features = get_image_feature(model_conv, img_filepath)
    entry = {
        'image_id'        : image_id,
        'category'        : category,
        'img_features'    : img_features,
        'attributes'      : attributes}
    return entry


def _load_dataset(model_conv, name):
    entries = []
    # let's only train on 8 images
    img_cnt = 0
    with open(base_path + 'attr_info.csv') as f:
        next(f)  # skip the header
        for line in f:
            if img_cnt > 7:
                break
            img_filepath = line.split(',')[0]
            attributes = line.split(',')[1:]
            entries.append(_create_entry(model_conv, img_filepath, attributes))
            img_cnt += 1
    return entries


class ImageRetrievalDataset(Dataset):
    def __init__(self, name, model_conv):
        super(ImageRetrievalDataset, self).__init__()
        assert name in ['train', 'val']
        self.num_ans_candidates = 1000
        self.entries = _load_dataset(model_conv, name)
        self.v_dim = 4096  # switch from hard-coded. img_feature is [1x4096] dim.

    def __getitem__(self, index):
        entry = self.entries[index]
        category = entry['category']
        img_features = entry['img_features']
        attributes = entry['attributes']
        target = torch.zeros(self.num_ans_candidates)
        # TODO: Correct answers during training
        # if labels is not None:
        #     target.scatter_(0, labels, scores)
        return torch.tensor(category), img_features.data, attributes, target

    def __len__(self):
        return len(self.entries)
