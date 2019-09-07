import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from torch.autograd import Variable
import pdb


# TODO: Change loss to triplet loss
def triplet_loss(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)

    # anchor --> top
    # positive --> another item from same id or similar attributes
    # negative --> same class but different id or dissimilar attributes

    distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
    distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
    loss = F.relu(distance_positive - distance_negative + margin)
    loss = loss.mean() if size_average else losses.sum()
    return loss


def compute_score_with_logits(predicted, target):
    # predicted.data
    # TODO double check that summed across first dimension.
    return sum(predicted * target)


def train(model, train_loader, eval_loader, num_epochs, output):
    utils.create_dir(output)
    optim = torch.optim.SGD(model.parameters(), lr=0.001)
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()
        print("epoch = ", epoch)

        for i, (category, feature, target_attributes) in enumerate(train_loader):
            print("    i = ", i)
            feature = Variable(feature).cuda()
            target_attributes = Variable(target_attributes).cuda()

            print(".")
            pred = model(feature, target_attributes)
            print("..")
            loss = triplet_loss(pred, target_attributes)
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
        eval_score, bound = evaluate(model, eval_loader)
        model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score

def evaluate(model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0
    for _, feature, target_attributes in iter(dataloader):
        feature = Variable(feature, volatile=True).cuda()
        pred = model(feature, None)
        batch_score = compute_score_with_logits(pred, target_attributes.cuda()).sum()
        score += batch_score
        upper_bound += (a.max(1)[0]).sum()  # TODO: change since multiple attributes can be the correct answer.
        num_data += pred.size(0)

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    return score, upper_bound
