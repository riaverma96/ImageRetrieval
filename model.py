from __future__ import print_function

import torch
import torch.nn as nn

from torch.nn.utils.weight_norm import weight_norm
import pdb

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


class Attention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(Attention, self).__init__()
        self.nonlinear = FCNet([v_dim + q_dim, num_hid])
        self.linear = weight_norm(nn.Linear(num_hid, v_dim), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, image_embedding, attribute):
        img_attribute = torch.cat((image_embedding, attribute), 1)  # [batch_size, (image_embedding.shape + attribute.shape)]
        joint_repr = self.nonlinear(img_attribute)  # [batch_size, num_hid]
        logits = self.linear(joint_repr)  # [batch_size, vdim]
        return logits


class BaseModel(nn.Module):
    def __init__(self, v_att, q_net, v_net, classifier):
        super(BaseModel, self).__init__()
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, image_features, attributes):
        """Forward

        image_features: [batch, image_feature_size] # image features size = 4096
        attributes: [batch, num_of_attributes]  # num of attributes = 463

        return: logits, not probs
        """
        # TODO: Do we need k feature representations per image? (UpDn has 1 per object).

        # attended image
        att = self.v_att(image_features, attributes)
        # print("att.shape = ", att.shape, "; (att * image_features) = ", (att * image_features))
        v_emb = (att * image_features)  # [batch, v_dim]  Note: removed .sum(1) bc 1 feature per image
        v_repr = self.v_net(v_emb)  # [batch, num_hid]

        # attribute
        q_repr = self.q_net(attributes)  # [batch, num_hid]

        joint_repr = q_repr * v_repr  # [batch, num_hid]
        logits = self.classifier(joint_repr)  # [batch, classes]
        return logits


def build_baseline(dataset, num_hid):
    v_att = Attention(dataset.v_dim, dataset.num_ans_candidates, num_hid)
    q_net = FCNet([dataset.num_ans_candidates, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel(v_att, q_net, v_net, classifier)
