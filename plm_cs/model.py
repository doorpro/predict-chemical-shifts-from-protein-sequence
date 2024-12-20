'''
coding:utf-8
@software:
@Time:2024/7/30 22:15
@Author:zhuhe
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import copy

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        # When batchsize is 1, comment out this line
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # mask = mask.unsqueeze(1)
        mask = mask.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(mask == 0, -1e9)
        # mask is an input tensor of the same size as scores, where zero or False replaces the corresponding position in scores with a large negative number

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out(concat)

        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def create_target_mask(padding_mask):
    '''Enter paddingmask, for example, the size is res*1, and output a res*res mask'''
    target_size = padding_mask.shape[1]
    batchsize = padding_mask.shape[0]
    # Create a lower triangular matrix that will be used as a forward mask
    look_ahead_mask = torch.tril(torch.ones(batchsize, target_size, target_size)) == 0
    target_mask = look_ahead_mask | ~padding_mask.unsqueeze(1)
    return ~target_mask

class Encoder1(nn.Module):
    def __init__(self,d_vec, d_model, heads, dropout):
        super().__init__()
        self.Linear0 = nn.Linear(d_vec, d_model)
        # Converts the input to 512 dimensions
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, d_model*4)
        self.fc2 = nn.Linear(d_model*4, d_model)

    def forward(self, x, mask):
        x = self.Linear0(x)
        x = self.pe(x)
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x2 = F.gelu(self.fc1(x2))
        x2 = self.fc2(x2)
        x = x + self.dropout_2(x2)
        return x

class Encoder2(nn.Module):
    def __init__(self, d_vec, d_model, heads, dropout):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 4*d_model)
        self.pre = nn.Linear(4*d_model, 1)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x = self.norm_2(x)
        x = self.dropout_3(F.gelu(self.fc(x)))
        emd = x
        x = self.pre(x)
        return x

class PLM_CS(nn.Module):
    def __init__(self, word_vec, d_model, heads, dropout):
        super().__init__()
        self.encoder = Encoder1(word_vec, d_model, heads, dropout)
        self.decoder = Encoder2(word_vec, d_model, heads, dropout)
    def forward(self, x, mask):
        x = self.encoder(x, mask)
        x = self.decoder(x, mask)
        return x