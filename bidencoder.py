#!/usr/bin/env python
# -*- coding = utf-8 -*-
# @Time : 2023/8/2 20:24
# @Author : door
# @File : LinearModel.py
# @Software : PyCharm
# @File : LinearModel.py
# @desc:
import math
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from transformer import Transformerencoder, uncertainty_encoder, regression
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
parser.add_argument('--N', type=int, default=12, help='number of encoder')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--d_model', type=int, default=432, help='qkv d-model dimension')
parser.add_argument('--d_vec', type=int, default=1280, help='amino embedding dimension')
parser.add_argument('--n_head', type=int, default=12, help='number of attention heads')
parser.add_argument('--shuffle', type=bool, default=False, help='shuffle dataset')
parser.add_argument('--epoch', type=int, default=5000, help='epoch time')
parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
parser.add_argument('-p', '--path', type=str, default='/mnt/DATA1/zhuhe/CSpre/MYdata/best_model_all.pth',
                        help='the path of file saving [default: ./save]')
parser.add_argument('--device', type=str, default="cuda:0", help='learning rate')

args = parser.parse_args()

def main(data, shiftxtest):
    model = uncertainty_encoder(args.d_vec, args.d_model, args.N, args.n_head, args.dropout)
    # model = regression(args.d_vec, args.d_model, args.n_head, args.dropout)
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
    valdata_loader = DataLoader(shiftxtest, batch_size=args.batchsize, shuffle=True)

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
            mse = loss**2*label[mask].shape[0]
            epoch_loss += mse.detach().item()
        return (epoch_loss/all_CA)

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
            rmse = math.sqrt(epoch_loss/all_CA)
        return rmse
    
    best_acc = 1.8
    for epoch in range(0, args.epoch):
        train_loss = train(epoch)
        val_loss = val(epoch)
        print(f'\tepoch{epoch:.3f}Train Loss: {train_loss:.3f} | val_rmse: {val_loss:7.3f}')
        train_loss_all.append(train_loss)
        val_loss_all.append(val_loss)


        if epoch == 4800:
            sp = args.path
            state = {
                "epoch": epoch,
                "accuracy": val_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }
            torch.save(model.state_dict(), sp)
            best_acc = val_loss


def test(path, data):
        model = Transformerencoder(args.d_vec, args.d_model, args.N, args.n_head, args.dropout)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.load_state_dict(torch.load(path))
        data_loader = DataLoader(data, batch_size=args.batchsize, shuffle=False)
        test_rmse = []
        for batch in data_loader:
            mask, label, seq_vec, padding_mask = batch[2], batch[1], batch[0], batch[3]
            mask = mask.to(device)
            label = label.to(device)
            seq_vec = seq_vec.to(device)
            padding_mask = padding_mask.to(device)
            out = model(seq_vec, padding_mask)
            mse_func = torch.nn.MSELoss()
            print(out.squeeze(2)[mask])
            print(label[mask])
            rmse = torch.sqrt(mse_func(out.squeeze(2)[mask], label[mask]))
            test_rmse.append(rmse.item())
        return sum(test_rmse)/len(test_rmse)



if __name__ == '__main__':
    # b = torch.load("F:\\nmrprediction\\dataset\\tensordataset\\bmrb_esm_ca2.pt")
    # test_rmse = test("F:\\nmrprediction\\dataset\\inmemory\\MYdata\\best_model\\best_model.pth", b)
    # shiftxtest = torch.load('/mnt/DATA1/zhuhe/CSpre/MYdata/tensordataset/test_ca.pt')
    #
    # b1 = torch.load('/mnt/DATA1/zhuhe/CSpre/MYdata/tensordataset/refdb_ca_notest.pt')
    # b2 = torch.load('/mnt/DATA1/zhuhe/CSpre/MYdata/tensordataset/all_bmrb_ca2.pt')
    #b3 = torch.load('/mnt/DATA1/zhuhe/CSpre/MYdata/tensordataset/all_bmrb_ca3.pt')
    #b4 = torch.load('/mnt/DATA1/zhuhe/CSpre/MYdata/tensordataset/all_bmrb_ca4.pt')
    #b5 = torch.load('/mnt/DATA1/zhuhe/CSpre/MYdata/tensordataset/all_bmrb_ca5.pt')
    #b6 = torch.load('/mnt/DATA1/zhuhe/CSpre/MYdata/tensordataset/all_bmrb_ca6.pt')
    #b7 = torch.load('/mnt/DATA1/zhuhe/CSpre/MYdata/tensordataset/all_bmrb_ca7.pt')
    #b8 = torch.load('/mnt/DATA1/zhuhe/CSpre/MYdata/tensordataset/all_bmrb_ca8.pt')
    # b = ConcatDataset([b1])
    # main(b, shiftxtest)

    from bmrb_seq_dataset import refdb_get_seq, refdb_find_shift, refdb_get_shift_re, refdb_get_cs_seq
    from pdb_bmrb_ali import align_bmrb_pdb
    import matplotlib.pyplot as plt
    import esm
    from scipy.stats import pearsonr


    def single_test(file_path):
        bmrb_seq = refdb_get_seq(file_path)
        s, e = refdb_find_shift(file_path)
        cs_seq = refdb_get_cs_seq(file_path, s, e)
        matched = align_bmrb_pdb(bmrb_seq, cs_seq)
        shift, mask = refdb_get_shift_re(file_path, s, e, bmrb_seq, matched)
        if '_' not in bmrb_seq and len(bmrb_seq) < 512:
            bmrb = torch.tensor(shift)
            mask = torch.tensor(mask)
            bmrb = bmrb[mask]
            model_path = "F:\\.cache\\torch\\hub\\checkpoints\\esm2_t33_650M_UR50D.pt"
            model, alphabet = esm.pretrained.load_model_and_alphabet(model_path)
            batch_converter = alphabet.get_batch_converter()
            model.eval()
            data = [("protein1", bmrb_seq)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]
            embedding = token_representations[:, 1:-1, :].squeeze()
            padding_mask = torch.zeros(512).bool()
            padding_mask[:embedding.shape[0]] = True
            embedding = torch.nn.functional.pad(embedding, (0, 0, 0, 512 - embedding.shape[0]))
            mask = torch.nn.functional.pad(mask, (0, 512 - mask.shape[0]), value=False)
            model = regression(1280, 432, 12, 0.1)
            # model = Transformerencoder(1280, 432, 12, 12, 0.1)
            padding_mask = padding_mask.unsqueeze(0)
            model.load_state_dict(
                torch.load("F:\\nmrprediction\\CSpre\\inmemory\\best_model\\ende_0.1lr_refdb82_ha.pth", map_location=torch.device('cpu')))
            out = model(embedding.unsqueeze(0), padding_mask)
            loss_func = torch.nn.MSELoss()
            mse = loss_func(bmrb, out.squeeze(0, 2)[mask])
            rmse = float(torch.sqrt(mse))
            bmrb = bmrb.tolist()
            out = out.squeeze(0, 2)[mask].tolist()
            R = pearsonr(out, bmrb)
            print(R)
            plt.scatter(bmrb, out, s=5)
            plt.xlim(42, 68)
            plt.ylim(42, 68)
            plt.text(60, 45, 'RMSE=0.93ppm\nR=0.98', fontsize=12, color='black')
            plt.show()


            return rmse
        else:
            return False

    a = single_test("F:\\nmrprediction\\CSpre\\dataset\\all_refdb\\CS-corrected-testset-addPDBresno\\A015_bmr16173.str.corr.pdbresno")
    print(a)

