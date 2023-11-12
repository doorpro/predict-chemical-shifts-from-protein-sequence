#!/usr/bin/env python
# -*- coding = utf-8 -*-
# @Time : 2023/7/1 10:56
# @Author : door
# @File : transformer.py
# @Software : PyCharm
# @File : transformer.py
# @desc:

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
        # 测试时因为batchsize为1，需将此行注释掉
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
        # mask为一个输入的tensor，大小与scores相同，其中为零或者为False的位置将scores中对应位置替换为大负数

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

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, 4*d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.Linear0 = nn.Linear(vocab_size, d_model)
        # 将输入转为512的维度
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        # N即为模型包含N层encoder
        self.norm = Norm(d_model)
    def forward(self, src, mask):
        x = self.Linear0(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)

class Transformerencoder(nn.Module):
    def __init__(self, word_vec, d_model, N, heads, dropout):
        # 输入的word_vec会经过一个embedding层后转为d_model的维度
        super().__init__()
        self.encoder = Encoder(word_vec, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(p = 0.1)
    def forward(self, word_vec, padding_mask):
        e_output = self.encoder(word_vec, padding_mask)
        e_output = self.out(e_output)
        # e_output = self.dropout(e_output)
        return e_output

class uncertainty_encoder(nn.Module):
    def __init__(self, word_vec, d_model, N, heads, dropout):
        # 输入的word_vec会经过一个embedding层后转为d_model的维度
        super().__init__()
        self.encoder = Encoder(word_vec, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, 1)
        self.sigma = nn.Linear(d_model, 1)
    def forward(self, word_vec, padding_mask):
        e_outputs = self.encoder(word_vec, padding_mask)
        cs = self.out(e_outputs)
        log_sigma = self.sigma(e_outputs)
        return cs, log_sigma

class encoder_outlay(nn.Module):
    def __init__(self, word_vec, d_model, N, heads, dropout):
        # 输入的word_vec会经过一个embedding层后转为d_model的维度
        super().__init__()
        self.encoder = Encoder(word_vec, d_model, N, heads, dropout)
        self.out1 = nn.Linear(d_model, d_model)
        self.out2 = nn.Linear(d_model, d_model)
        self.out3 = nn.Linear(d_model, 1)
    def forward(self, word_vec, padding_mask):
        e_outputs = self.encoder(word_vec, padding_mask)
        output = F.relu(self.out1(e_outputs))
        output = F.tanh(self.out2(output))
        output = self.out3(output)
        return output

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.linear = nn.Linear(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.linear(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, 4*d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs,
                                           src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output

def create_target_mask(padding_mask):
    '''输入paddingmask,例如大小为res*1，输出一个res*res的mask'''
    target_size = padding_mask.shape[1]
    batchsize = padding_mask.shape[0]
    # 创建一个下三角矩阵，它将用作前瞻遮罩
    look_ahead_mask = torch.tril(torch.ones(batchsize, target_size, target_size)) == 0
    target_mask = look_ahead_mask | ~padding_mask.unsqueeze(1)
    return ~target_mask

class Encoder2(nn.Module):
    def __init__(self,d_vec, d_model, heads, dropout):
        super().__init__()
        self.Linear0 = nn.Linear(d_vec, d_model)
        # 将输入转为512的维度
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

class Decoder2(nn.Module):
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
        x = self.pre(x)
        return x

class regression(nn.Module):
    def __init__(self, word_vec, d_model, heads, dropout):
        super().__init__()
        self.encoder = Encoder2(word_vec, d_model, heads, dropout)
        self.decoder = Decoder2(word_vec, d_model, heads, dropout)
    def forward(self, x, mask):
        x = self.encoder(x, mask)
        x = self.decoder(x, mask)
        return x

class LSTM(nn.Module):
    def __init__(self, word_vec, hidden_size, num_layers, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=word_vec, out_channels=32, kernel_size=129, padding=64)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=257, padding=128)
        self.biLSTM = nn.LSTM(input_size=1312, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.norm = Norm(1312)
        self.dropout = nn.Dropout(dropout)
        self.Linear1 = nn.Linear(1280, 32)
        self.Linear2 = nn.Linear(2048, 1)
    def forward(self, x):
        x2 = self.dropout(F.relu(self.conv1(x.permute(0, 2, 1))))
        x2 = self.dropout(F.relu(self.conv2(x2)))
        x = torch.concat((x, x2.permute(0, 2, 1)), dim=2)
        x = self.norm(x)
        x, _ = self.biLSTM(x)
        x = self.dropout(x)
        x = self.Linear2(x)
        return x
