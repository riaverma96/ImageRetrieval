import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from torch.autograd import Variable
from random import sample
import numpy
import pdb


# TODO: Differentiate based on outfit id as well? (As opposed to just img id).
def triplet_loss(logits, labels, img_id, enumerated_ids, attr):
    assert logits.dim() == 2

    # Triplet loss
    losses = 0
    for i in range(logits.size(0)):
        pos_instance, neg_instance = None, None
        while True:  # negative sample
            sampled_instance = sample(enumerated_ids, 1)[0]
            sampled_img_id = sampled_instance[0]
            if sampled_img_id != img_id[i]:
                neg_instance = sampled_instance
                break

        for instance in enumerated_ids:  # positive sample
            candidate_img_id = instance[0]
            if candidate_img_id == img_id[i]:
                pos_instance = instance
                break

        anchor = logits[i]
        positive = torch.tensor(attr[pos_instance[0]]).cuda()  # (1964, 2, 'front')
        negative = torch.tensor(attr[neg_instance[0]]).cuda()

        distance_positive = torch.dot(anchor, positive)
        distance_negative = torch.dot(anchor, negative)

        margin = Variable(torch.tensor(0.1)).cuda()
        losses += F.relu(distance_positive - distance_negative + margin)

    losses = losses.mean() # or losses.sum()
    return losses


def compute_score_with_logits(predicted, target):
    # predicted.data
    # TODO double check that summed across first dimension.
    return sum(predicted * target)


def train(model, train_loader, eval_loader, num_epochs, output, train_enumerated_ids, train_attr, eval_enumerated_ids, eval_attr):
    utils.create_dir(output)
    optim = torch.optim.SGD(model.parameters(), lr=0.001)
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()
        print("epoch = ", epoch)

        for i, data in enumerate(train_loader):
            feature = data['feature']
            target_attributes = data['target_attributes']
            img_id = data['img_id']
            outfit_id = data['outfit_id']
            shot_type = data['shot_type']

            feature = Variable(feature).cuda()
            target_attributes = Variable(target_attributes).cuda()

            pred = model(feature, target_attributes)  # (512, 463)
            loss = triplet_loss(pred, target_attributes, img_id, train_enumerated_ids, train_attr)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            # batch_score = compute_score_with_logits(pred, target.data).sum()
            v_dim = 4096  # switch from hard-coded. img_feature is [1x4096] dim.
            total_loss += loss.data[0] * v_dim  # v.size(0)
            # train_score += batch_score

        total_loss /= len(train_loader.dataset)
        # train_score = 100 * train_score / len(train_loader.dataset)
        train_score = 0
        model.train(False)
        eval_score, bound = evaluate(model, eval_loader, eval_enumerated_ids, eval_attr)
        model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        # logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score

def evaluate(model, dataloader, eval_enumerated_ids, eval_attr):
    score = 0
    upper_bound = 0
    return score, upper_bound

    # TODO: fix evaluate
    num_data = 0
    for feature, target_attributes, img_id, outfit_id, shot_type in iter(dataloader):
        feature = Variable(feature, volatile=True).cuda()
        pred = model(feature, None)
        batch_score = compute_score_with_logits(pred, target_attributes.cuda()).sum()
        score += batch_score
        upper_bound += (a.max(1)[0]).sum()  # TODO: change since multiple attributes can be the correct answer.
        num_data += pred.size(0)

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    return score, upper_bound
