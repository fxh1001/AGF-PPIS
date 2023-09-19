import torch
import torch.nn as nn
import torch.distributed as dist

class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out

class Cal_Contrastive_loss(nn.Module):
    def __init__(self,batch_size,temperature,word_size):
        super(Cal_Contrastive_loss, self).__init__()
        self.batch_size =batch_size
        self.temperature =temperature
        self.word_size = word_size
        self.mask = self.mask_correlated_samples(batch_size,word_size)
        self.loss_func = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self,batch_size,word_size):
        N = 2 * batch_size * word_size
        mask = torch.ones((N,N),dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * word_size):
            mask[i,batch_size+i] = 0
            mask[batch_size+i,i] = 0

        return mask

    def forward(self,z_i,z_j):
        z_i = torch.squeeze(z_i)
        z_j = torch.squeeze(z_j)

        N = 2 * self.batch_size * self.word_size
        z = torch.cat((z_i,z_j),dim=0)
        if self.word_size > 1:
            z = torch.cat(GatherLayer.apply(z),dim=0)
        sim = self.similarity_f(z.unsqueeze(1),z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim,self.batch_size * self.word_size)
        sim_j_i = torch.diag(sim,self.batch_size * self.word_size)

        positive_sample = torch.cat((sim_i_j,sim_j_i),dim=0).reshape(N,1)
        negeative_sample = sim[self.mask].reshape(N,-1)

        label = torch.zeros(N).to(positive_sample.device).long()
        logist = torch.cat((positive_sample,negeative_sample),dim=-1)
        loss = self.loss_func(logist,label)
        loss /= N
        return loss