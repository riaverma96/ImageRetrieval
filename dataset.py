from __future__ import print_function
import os
import re
import json
import cPickle
import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset
from img_to_vec import Img2Vec


base_path = "./data/"


def process_attributes(base_path=None, old_filename=None, new_filename=None):
    """
    Processes attribute file to be CSV compatible. Saves in new file in the same directory.
    """
    attr = pd.read_csv(base_path + 'list_attr_img.txt', skiprows=2, sep='\s+', names=['image_name'] + ['attr_%d' % i for i in range(1000)])
    attr.replace(-1, 0, inplace=True)
    attr.to_csv(base_path + 'attr_info.csv', index=False)


def get_img_fields(img_name):
    """
    Gets image category and image id from image file path. Expects file path to
    be in following format:  img/<category>/img_<img_id>.jpg
    """
    dir = img_names.split('/')
    category = dir[1]
    image_id = dir[-1].split('.')[0].split('_')[1]
    return category, image_id


def get_image_feature(img2vec, img_filepath):
    img = Image.open(base_path + img_filepath)
    return img2vec.get_vec(img)


# TODO: Complete
def _create_entry(img2vec, img_filepath):
    category, image_id = get_img_fields(img_filepath)
    img_features = get_image_feature(img2vec)
    attributes = None
    entry = {
        'image_id'        : image_id,
        'category'        : category,
        'img_features'    : img_features,
        'attributes'      : attributes}
    return entry


def _load_dataset(img2vec, dataroot, name, img_id2val):
    entries = []
    for question, answer in zip(questions, answers):
        entries.append(_create_entry(img2vec, img))
    return entries


class ImageRetrievalDataset(Dataset):
    def __init__(self, name, dataroot='data'):
        super(ImageRetrievalDataset, self).__init__()
        assert name in ['train', 'val']
        self.num_ans_candidates = 1000
        img2vec = Img2Vec(cuda=True, model='resnet-18')
        self.entries = _load_dataset(img2vec)

    def __getitem__(self, index):
        entry = self.entries[index]
        img_features = entry['img_features']
        attributes = entry['attributes']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)
        return img_features, attributes, target

    def __len__(self):
        return len(self.entries)
