import argparse
import os
import json

import numpy as np
import pandas as pd
import pdb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import ImageRetrievalDataset
from train import train
import model
import torchvision

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--image_filenames', type=str, default='./data/img/Sheer_Pleated-Front_Blouse/img_00000001.jpg', help='filepath to raw images.')
    parser.add_argument('--attribute_filename', type=str, default='./data/info.csv', help='filepath to attributes.')
    parser.add_argument('--output', type=str, default='saved_models/exp0')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    model_conv = torchvision.models.vgg19(pretrained='imagenet')
    # convert all the layers to list and remove the last one
    features = list(model_conv.classifier.children())[:-1]
    # convert it into container and add it to our model class.
    model_conv.classifier = nn.Sequential(*features)

    train_dset = ImageRetrievalDataset('train', model_conv)
    eval_dset = ImageRetrievalDataset('val', model_conv)
    constructor = 'build_baseline'
    model = getattr(model, constructor)(train_dset, args.num_hid)  #.cuda()
    model = nn.DataParallel(model).cuda()

    train_loader = DataLoader(train_dset, args.batch_size, shuffle=True, num_workers=1)
    eval_loader =  DataLoader(eval_dset, args.batch_size, shuffle=True, num_workers=1)
    print("training ...")
    train(model, train_loader, eval_loader, args.epochs, args.output,  \
        train_dset.enumerated_ids, train_dset.attributes, eval_dset.enumerated_ids, eval_dset.attributes)
