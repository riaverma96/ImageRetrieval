import argparse
import os
import json

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
import pdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--image_filenames', type=str, default='./data/img/Sheer_Pleated-Front_Blouse/img_00000001.jpg', help='filepath to raw images.')
    parser.add_argument('--attribute_filename', type=str, default='./data/info.csv', help='filepath to attributes.')
    parser.add_argument('--output', type=str, default='saved_models/exp0')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    train_dset = ImageRetrievalDataset('train')
    val_dset = ImageRetrievalDataset('val')
    constructor = 'build_baseline'
    model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
    model = nn.DataParallel(model).cuda()

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1)
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1)
    train(model, train_loader, eval_loader, args.epochs, args.output)
