#-*-coding:utf-8-*-
import numpy as np
import torch
import torch.nn.functional as F
def calc_label_sim(label_1, label_2):
    Sim = label_1.float().mm(label_2.float().t())
    return Sim


def discrimination_loss(x, y, alpha, beta):
    cos = lambda x, y: x.mm(y.t()) / ((x ** 2).sum(1, keepdim=True).sqrt().mm((y ** 2).sum(1, keepdim=True).sqrt().t())).clamp(min=1e-6) * 2.

    theta12 = cos(x, y)
    theta11 = cos(x, x)
    theta22 = cos(y, y)

    loss1 = ((theta11 - theta22)**2).mean()   # 模态内
    loss2 = ((theta12-theta12.t())**2).mean() # 模态间
    loss3 = F.mse_loss(x,y)                   # MSE
    return  alpha * (loss1 + loss2) + beta * loss3

def gce_loss(pred, labels, q):
    pred = F.softmax(pred, dim=1)

    mae = (1. - torch.sum(labels.float() * pred, dim=1)**q).div(q)

    return mae.mean()
def feature_augmentation(features, labels, gamma1):
    lam = gamma1
    index = np.arange(features.shape[0])
    np.random.shuffle(index)
    random_features = features[index]
    random_labels = labels[index]
    new_features = lam * features + (1 - lam) * random_features
    new_labels =  lam * labels + (1 - lam)* random_labels
    return new_features, new_labels
class LossModule(torch.nn.Module):
    def __init__(self, W, alpha, beta, eta, gamma, **kwargs):
        torch.nn.Module.__init__(self)
        self.L = torch.pinverse(W)
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.gamma = gamma
    def forward(self, features, labels, predicts, epoch):
        label_features = torch.mm(labels, self.L)
        term1 = discrimination_loss(label_features, features, self.alpha, self.beta)
        q = min(1., 0.01 * epoch)
        term2 = gce_loss(predicts, labels, q)
        return term1 + self.eta * term2
