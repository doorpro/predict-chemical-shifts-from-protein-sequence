'''
coding:utf-8
@software:
@Time:2024/7/31 1:15
@Author:door
'''
import math
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from model import regression
from torch.utils.data import random_split
import argparse
import numpy as np

seed = 42
torch.manual_seed(seed)
# 为CPU设置随机种子
torch.cuda.manual_seed(seed)
# 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)
# if you are using multi-GPU，为所有GPU设置随机种子
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()

parser.add_argument('--batchsize', type=int, default=16, help='Batch size for training')
parser.add_argument('--N', type=int, default=6, help='number of encoder')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--d_model', type=int, default=512, help='qkv d-model dimension')
parser.add_argument('--d_vec', type=int, default=1280, help='amino embedding dimension')
parser.add_argument('--n_head', type=int, default=8, help='number of attention heads')
parser.add_argument('--shuffle', type=bool, default=False, help='shuffle dataset')
parser.add_argument('--epoch', type=int, default=5000, help='epoch time')
parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
parser.add_argument('-p', '--path', type=str, default='/mnt/DATA1/zhuhe/CSpre/MYdata/best_model_all.pth',
                        help='the path of file saving [default: ./save]')
parser.add_argument('--device', type=str, default="cuda:0", help='learning rate')
args = parser.parse_args()

def main(data, shiftxtest):
    # model = uncertainty_encoder(args.d_vec, args.d_model, args.N, args.n_head, args.dropout)
    model = regression(args.d_vec, args.d_model, args.n_head, args.dropout)
    # model = LinearModel(1280, 1)
    device = torch.device(args.device)
    train_loss_all = []
    val_loss_all = []
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    loss_func = torch.nn.MSELoss()
    model.to(device)
    def init_weights1(model):
        if isinstance(model, torch.nn.Linear):
            torch.nn.init.kaiming_uniform(model.weight)

    def init_weights_kaiming(model):
        if hasattr(model, 'weight') and model.weight.dim() > 1:
            torch.nn.init.kaiming_uniform(model.weight.data)

    def init_weights_xavier(model):
        if isinstance(model, torch.nn.MultiheadAttention):
            torch.nn.init.xavier_uniform_(model.in_proj_weight)
            torch.nn.init.xavier_uniform_(model.out_proj.weight)

    model.apply(init_weights1)
    train_size = int(len(data) * 1)
    val_size = len(data) - train_size
    train_dataset, val_dataset = random_split(data, [train_size, val_size])
    traindata_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
    valdata_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=True)

    def train(epoch):
        model.train()
        epoch_loss = 0
        all_CA = 0
        for i, batch in enumerate(traindata_loader):
            mask, label, seq_vec, padding_mask = batch[2], batch[1], batch[0], batch[3]
            mask = mask.to(device)
            label = label.to(device)
            seq_vec = seq_vec.to(device)
            padding_mask = padding_mask.to(device)
            out = model(seq_vec, padding_mask)
            loss = torch.sqrt(loss_func(out.squeeze(2)[mask], label[mask]))
            # out, log_sigma = model(seq_vec, padding_mask)
            # sigma_normal = torch.sqrt(torch.mean(0.5*(torch.exp((-1)*log_sigma)) * (out.squeeze(2)[mask] - label[mask])**2 + 0.5*log_sigma))
            all_CA += label[mask].shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mse = loss ** 2 * label[mask].shape[0]
            epoch_loss += mse.detach().item()
        return (epoch_loss / all_CA)

    def val(epoch):
        # model.train()
        epoch_loss = 0
        all_CA = 0
        for i, batch in enumerate(valdata_loader):
            mask, label, seq_vec, padding_mask = batch[2], batch[1], batch[0], batch[3]
            mask = mask.to(device)
            label = label.to(device)
            seq_vec = seq_vec.to(device)
            padding_mask = padding_mask.to(device)
            out = model(seq_vec, padding_mask)
            loss = loss_func(out.squeeze(2)[mask], label[mask])
            all_CA += label[mask].shape[0]
            loss = loss * label[mask].shape[0]
            epoch_loss += loss.item()
            rmse = math.sqrt(epoch_loss / all_CA)
        return rmse

    best_acc = 1.8
    for epoch in range(0, args.epoch):
        train_loss = train(epoch)
        val_loss = val(epoch)
        print(f'\tepoch{epoch:.3f}Train Loss: {train_loss:.3f} | val_rmse: {val_loss:7.3f}')
        train_loss_all.append(train_loss)
        val_loss_all.append(val_loss)

        if val_loss<best_acc:
            sp = args.path
            state = {
                "epoch": epoch,
                "accuracy": val_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }
            torch.save(state, sp)
            best_acc = val_loss

if __name__ == '__main__':
    print(True)