import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from FocalLoss import FocalLoss
LEARNING_RATE = 0.001
def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias,0)

class FeedForwardNetwork(nn.Module):
    def __init__(self,hidden_size,fliter_size,dropout_rate):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(hidden_size,fliter_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.layer2 = nn.Linear(fliter_size,hidden_size)

        initialize_weight(self.layer1)
        initialize_weight(self.layer2)

    def forward(self,x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)

        return x

class MuliHeadAttention(nn.Module):
    def __init__(self,hidden_size,dropout_rate,head_size = 6):
        super(MuliHeadAttention, self).__init__()
        self.head_size = head_size
        self.attn_size = attn_size = hidden_size // head_size
        self.scale = attn_size ** -0.5
        self.linner_q = nn.Linear(hidden_size,head_size*attn_size,bias=False)
        self.linner_k = nn.Linear(hidden_size,head_size*attn_size,bias=False)
        self.linner_v = nn.Linear(hidden_size,head_size*attn_size,bias=False)

        initialize_weight(self.linner_q)
        initialize_weight(self.linner_k)
        initialize_weight(self.linner_v)

        self.attn_dropout = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(head_size*attn_size,hidden_size,bias=False)
        initialize_weight(self.output_layer)

    def forward(self,q,k,v,mask,adj_matrix):
        origin_q_size = q.size()
        batch_size = q.size(0)

        d_k = self.attn_size
        d_v = self.attn_size

        q = self.linner_q(q).view(batch_size,-1,self.head_size,d_k)
        k = self.linner_k(k).view(batch_size,-1,self.head_size,d_k)
        v = self.linner_v(v).view(batch_size,-1,self.head_size,d_k)

        q = q.transpose(1,2)
        v = v.transpose(1,2)
        k = k.transpose(1,2).transpose(2,3)

        q.mul_(self.scale)

        x = torch.matmul(q,k)
        x = torch.mul(adj_matrix,x)
        x.masked_fill_(mask.unsqueeze(1),-1e9)
        x = torch.softmax(x,dim=-1)
        x = self.attn_dropout(x)
        x = x.matmul(v)
        x = x.transpose(1,2).contiguous()
        x = x.view(batch_size,-1,self.head_size*d_v)
        x = self.output_layer(x)
        assert x.size() == origin_q_size

        return x

class GCN(nn.Module):
    def __init__(self,hidden_size):
        super(GCN, self).__init__()
        self.hidden_size = hidden_size
        self.weight = Parameter(torch.FloatTensor(hidden_size,hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_size)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self,x,adj):
        y = torch.matmul(adj,x)
        y = torch.matmul(y,self.weight)

        return y

class BasicBlock(nn.Module):
    def __init__(self,hidden_size,flier_size,dropout_rate):
        super(BasicBlock, self).__init__()
        self.relu = nn.ReLU()

        self.attn_norm = nn.LayerNorm(hidden_size,eps=1e-6)
        self.MultiAttention = MuliHeadAttention(hidden_size,dropout_rate)
        self.attn_dropout = nn.Dropout(dropout_rate)

        self.gcn_norm = nn.LayerNorm(hidden_size,eps=1e-6)
        self.GCN = GCN(hidden_size)
        self.gcn_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size,eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size,flier_size,dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self,x,mask,adj_matrix,adj):
        y = self.attn_norm(x)
        y = self.relu(y)
        y = self.MultiAttention(y,y,y,mask,adj_matrix)
        y = self.attn_dropout(y)
        x = x + y

        y = self.gcn_norm(x)
        y = self.relu(y)
        y = self.GCN(y,adj)
        y = self.gcn_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.relu(y)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y

        return x

class DeepAttentionGCN(nn.Module):
    def __init__(self,input_size,hidden_size,fliter_size,output_size,dropout_rate,n_layers):
        super(DeepAttentionGCN, self).__init__()
        self.input_linner = nn.Linear(input_size,hidden_size)
        encoders = [BasicBlock(hidden_size,fliter_size,dropout_rate) for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.last_norm = nn.LayerNorm(hidden_size,eps=1e-6)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.output_linner = nn.Linear(hidden_size,output_size)

        initialize_weight(self.input_linner)

    def forward(self,input,mask,adj_matrix,adj,contrastive = False):
        x = self.input_linner(input)
        for i ,layers in enumerate(self.layers):
            x = layers(x,mask,adj_matrix,adj)

            if contrastive:
                if i < 5:
                    random_noise = torch.rand_like(x).to(x.device)
                    x = x + torch.sign(x)*F.normalize(random_noise,dim=-1)*0.1

        feature_map = x
        if not contrastive:
            x = self.last_norm(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.output_linner(x)

        return x,feature_map

class FinalModel(nn.Module):
    def __init__(self,input_size,hidden_size,fliter_size,output_size,dropout_rate,n_layers,contrastive=False,alpha=None):
        super(FinalModel, self).__init__()
        self.constrative = contrastive
        self.model = DeepAttentionGCN(input_size,hidden_size,fliter_size,output_size,dropout_rate,n_layers)
        self.optimizer = torch.optim.Adam(self.parameters(),lr=LEARNING_RATE)
        self.loss_func = FocalLoss(alpha=alpha)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.6, patience=10,
                                                                    min_lr=1e-6)

    def forward(self,input,mask,adj_matrix,adj):
        x = input.float()
        output,feature_map = self.model(x,mask,adj_matrix,adj)
        if self.constrative:
            _,feature_map_noise = self.model(x,mask,adj_matrix,adj,contrastive = True)

            return output,feature_map,feature_map_noise

        return output




