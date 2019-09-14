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

        pdb.set_trace()
        anchor = logits[i]
        positive = attr[pos_instance[i]]
        negative = attr[neg_instance[i]]

        distance_positive = torch.dot(anchor, positive)
        distance_negative = torch.dot(anchor, negative)

        pdb.set_trace()
        margin = Variable(torch.tensor(0.1)).cuda()
        losses += F.relu(distance_positive - distance_negative + margin)
    losses = loss.mean() if size_average else losses.sum()

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

        for i, (feature, target_attributes, img_id, outfit_id, shot_type) in enumerate(train_loader):
            print("    i = ", i)
            feature = Variable(feature).cuda()
            target_attributes = Variable(target_attributes).cuda()

            pred = model(feature, target_attributes)
            loss = triplet_loss(pred, target_attributes, img_id, train_enumerated_ids, train_attr)
            print("triplet loss = ", loss)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.data[0] * v.size(0)
            train_score += batch_score

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        model.train(False)
        eval_score, bound = evaluate(model, eval_loader, eval_enumerated_ids, eval_attr)
        model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score

def evaluate(model, dataloader, eval_enumerated_ids, eval_attr):
    score = 0
    upper_bound = 0
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
