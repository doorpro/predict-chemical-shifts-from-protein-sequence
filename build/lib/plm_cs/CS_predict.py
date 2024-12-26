'''
!/usr/bin/env python
-*- coding = utf-8 -*-
@Time : 2024/8/7 20:24
@Author : ZhuHe
@File : LinearModel.py
@Software : PyCharm
@File : CS_predict.py
'''
import torch
from plm_cs.model import PLM_CS
import esm
import pandas as pd
import os
import argparse
import traceback
import json
from Bio import SeqIO

# current_file_directory = os.path.dirname(os.path.abspath(__file__))
# # os.chdir(current_file_directory)
# config_path = os.path.join(current_file_directory, 'config.json')
# with open(config_path, 'r') as f:
#     config = json.load(f)

config = {
    "model_paths": {
        "reg_HA": "./plm_cs/ckpt/model_ckpt/reg_ha.pth",
        "reg_H": "./plm_cs/ckpt/model_ckpt/reg_h.pth",
        "reg_N": "./plm_cs/ckpt/model_ckpt/reg_n.pth",
        "reg_CA": "./plm_cs/ckpt/model_ckpt/reg_ca.pth",
        "reg_CB": "./plm_cs/ckpt/model_ckpt/reg_cb.pth",
        "reg_C": "./plm_cs/ckpt/model_ckpt/reg_c.pth"
    }
}

amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']

def read_fasta(file_path):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(record.seq)
    return sequences

def predict_from_seq(protein_sequence, result_file):
        print("The first run takes time because the ESM weights need to be downloaded")
        cs_df = {'HA':[], 'H':[], 'N':[], 'CA':[], 'CB':[], 'C':[]}
        # model_path = config['model_paths']['esm_model']
        # model_path = ".\\esm_ckpt\\esm2_t33_650M_UR50D.pt"
        # model, alphabet = esm.pretrained.load_model_and_alphabet(model_path)

        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
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
                torch.load(config['model_paths'][f'reg_{atom}'], map_location=torch.device('cpu'), weights_only=True))
            # load the model
            model.eval()
            out = model(embedding.unsqueeze(0), padding_mask)
            chemical_shifts = out.squeeze(2).squeeze(0).detach().numpy()
            chemical_shifts = chemical_shifts[:len(protein_sequence)]
            cs_df[atom] = chemical_shifts


        cs_df = pd.DataFrame(cs_df)   
        cs_df.to_csv(result_file, index=False)
        
        print("The chemical shifts of the protein sequence have been saved in the result folder:"+result_file)

def main():
    parser = argparse.ArgumentParser(description="Predict chemical shifts from protein sequence.")
    parser.add_argument('input', type=str, help='Fasta file or Protein sequence')
    parser.add_argument('--result_file', type=str, default='./result/new.csv', help='Output CSV file for results')
    args = parser.parse_args()
    
    input_file_seq = args.input
    if os.path.exists(input_file_seq):
        sequences = read_fasta(input_file_seq)
        if len(sequences) > 1:
            raise ValueError("Multiple sequences detected in the input file, please input one sequence at a time")
        args.sequence = str(sequences[0])
    else:
        print("Input is not a file, assuming input is a protein sequence")
        args.sequence = str(input_file_seq)

    args.sequence = args.sequence.upper()
    if not args.sequence.isalpha():
        raise ValueError("Protein Sequence formatting error")
    if  [char for char in args.sequence if char not in amino_acids]:
        raise ValueError("Protein Sequence contains invalid characters, it is recommended to use X to replace unknown amino acids")
    try:
        predict_from_seq(args.sequence, args.result_file)
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        # with open("error.log", "w") as f:
        #     f.write(str(e))

if __name__ == '__main__':
    main()