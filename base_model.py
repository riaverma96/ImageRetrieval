from __future__ import print_function

import torch
import torch.nn as nn

from torch.nn.utils.weight_norm import weight_norm
from torchvision.models import vgg19

class FCNet(nn.Module):
    """
    Simple class for non-linear fully connect network
    """
    def __init__(self, dims):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            layers.append(nn.ReLU())
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        layers.append(nn.ReLU())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class BaseModel(nn.Module):
    def __init__(self, a_net, v_net, classifier):
        super(BaseModel, self).__init__()

        self.a_net = a_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, v, b, q, labels):
        """
        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]
        return: logits, not probs
        """
        q_repr = self.a_net(q)
        v_repr = self.v_net(v)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits

def build_baseline(dataset, num_hid):
    a_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])

    num_ans_candidates = 1000
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, num_ans_candidates, 0.5)
    return BaseModel(a_net, v_net, classifier)
