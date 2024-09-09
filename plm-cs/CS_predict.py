#!/usr/bin/env python
# -*- coding = utf-8 -*-
# @Time : 2024/8/7 20:24
# @Author : ZhuHe
# @File : LinearModel.py
# @Software : PyCharm
# @File : CS_predict.py
# @desc:
import torch
from model import PLM_CS
import esm
import pandas as pd
import os
import argparse
import traceback
import json

current_file_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_file_directory)
# change the current directory to the directory of this file
config_path = os.path.join(current_file_directory, 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

def predict_from_seq(protein_sequence, result_file):
        print("Starting prediction...")
        # 你的代码逻辑
        # 在关键步骤添加打印信息
        print("Loading model...")
        # 加载模型代码
        print("Model loaded.")
        # 其他代码逻辑
        print("Prediction completed.")
        cs_df = {'HA':[], 'H':[], 'N':[], 'CA':[], 'CB':[], 'C':[]}
        model_path = config['model_paths']['esm_model']
        # model_path = ".\\esm_ckpt\\esm2_t33_650M_UR50D.pt"
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_path)
        # Load ESM-2 650m model

        batch_converter = alphabet.get_batch_converter()
        model.eval()
        data = [("sequence", protein_sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        embedding = token_representations[:, 1:-1, :].squeeze()
        embedding = torch.nn.functional.pad(embedding, (0, 0, 0, 512 - embedding.shape[0]))
        # the esm embedding of the protein sequence

        padding_mask = torch.zeros(512).bool()
        padding_mask[:len(protein_sequence)] = True
        padding_mask = padding_mask.unsqueeze(0)

        model = PLM_CS(1280, 512, 8, 0.1)

        for atom in ['HA', 'H', 'N', 'CA', 'CB', 'C']:
            model.load_state_dict(
                torch.load(config['model_paths'][f'reg_{atom}'], map_location=torch.device('cpu')))
            # load the model
            model.eval()
            out = model(embedding.unsqueeze(0), padding_mask)
            chemical_shifts = out.squeeze(2).squeeze(0).detach().numpy()
            chemical_shifts = chemical_shifts[:len(protein_sequence)]
            cs_df[atom] = chemical_shifts


        cs_df = pd.DataFrame(cs_df)   
        cs_df.to_csv(result_file, index=False)
        
        print("The chemical shifts of the protein sequence have been saved in the result folder.")

def main():
    parser = argparse.ArgumentParser(description="Predict chemical shifts from protein sequence.")
    parser.add_argument('sequence', type=str, help='Protein sequence')
    parser.add_argument('--result_file', type=str, default='./result/new.csv', help='Output CSV file for results')
    args = parser.parse_args()
    
    try:
        predict_from_seq(args.sequence, args.result_file)
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        with open("error.log", "w") as f:
            f.write(str(e))

if __name__ == '__main__':
    main()