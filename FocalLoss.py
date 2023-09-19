import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import  Variable
class FocalLoss(nn.Module):
    def __init__(self,class_num=2,alpha = None,gamma=2,size_average = True):
        super(FocalLoss, self).__init__()
        if alpha == None:
            self.alpha = Variable(torch.ones(class_num,1))
        else:
            if isinstance(alpha,Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.size_average = size_average

    def forward(self,inputs,label):
        N = inputs.size(0)
        C = inputs.size(1)
        P  = F.softmax(inputs,dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        targets = label.view(-1,1)
        class_mask.scatter_(1,targets,1.)

        self.alpha = self.alpha.cuda()
        alpha_ = self.alpha[targets.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)
        log_p = probs.log()

        batch_loss = -alpha_*(torch.pow((1-probs),self.gamma))*log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss