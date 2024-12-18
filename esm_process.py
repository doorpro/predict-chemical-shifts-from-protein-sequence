'''
coding:utf-8
@software:
@Time:2024/7/31 0:23
@Author:door
'''
import torch
import esm
from torch.utils.data import TensorDataset

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()
'''In the data processing process, the esm model is used in advance to convert the sequence to embedding'''

def esm_process(protein_sequence, chemical_shifts, mask):
    data = [("protein1", protein_sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    embedding = token_representations[:, 1:-1, :].squeeze()
    esm_vec = torch.nn.functional.pad(embedding, (0, 0, 0, 512 - embedding.shape[0]))
    # padding the size of tensor from "res*1280" to 512*1280
    mask = torch.tensor(mask)
    mask = torch.nn.functional.pad(mask, (0, 512-mask.shape[0]), value=False)
    label = torch.tensor(chemical_shifts)
    label = torch.nn.functional.pad(label, (0, 512-label.shape[0]))
    padding_mask = torch.zeros(512).bool()
    padding_mask[:len(protein_sequence)] = True
    padding_mask = padding_mask.unsqueeze(0)
    return esm_vec, label, mask, padding_mask

def main(refdb_path, save_path, atom_type):
    all_esm_vec = torch.zeros(1, 512, 1280)
    all_label = torch.zeros((1, 512))
    all_mask = torch.zeros((1, 512)).bool()
    all_padding_mask = torch.zeros((1, 512)).bool()
    for root, directories, files in os.walk(refdb_path):
        for file in files:
            file_path =str(file.split(".")[0])
            bmrb_seq_list = extract_protein_sequence(file_path)
            s, e = refdb_find_shift(file_path)
            cs_seq = refdb_get_cs_seq(file_path, s, e)
            matched = align_bmrb_pdb(bmrb_seq, cs_seq)
            shift, mask = refdb_get_shift_re(file_path, s, e, bmrb_seq, matched, atom_type)
            for i, bmrb_seq in enumerate(bmrb_seq_list):
                if '_' not in bmrb_seq and 0<len(bmrb_seq) < 512:
                    data = [("protein1", bmrb_seq_list[i])]
                    batch_labels, batch_strs, batch_tokens = batch_converter(data)
                    with torch.no_grad():
                        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
                    token_representations = results["representations"][33]
                    embedding = token_representations[:, 1:-1, :].squeeze()
                    embedding = torch.nn.functional.pad(embedding, (0, 0, 0, 512 - embedding.shape[0]))
                    # padding the size of tensor from "res*1280" to 512*1280
                    label = torch.tensor(shift[i])
                    padding_mask = torch.zeros(512).bool()
                    padding_mask[:label.shape[0]] = True
                    label = torch.nn.functional.pad(label, (0, 512-label.shape[0]))
                    # padding the size of tensor from "res" to 512
                    mask = torch.tensor(mask[i])
                    mask = torch.nn.functional.pad(mask, (0, 512-mask.shape[0]), value=False)
                    if not torch.all(mask.eq(False)):
                        all_esm_vec = torch.cat((all_esm_vec, embedding.unsqueeze(0)), dim=0)
                        all_label = torch.cat((all_label, label.unsqueeze(0)), dim=0)
                        all_mask = torch.cat((all_mask, mask.unsqueeze(0)), dim=0)
                        all_padding_mask = torch.cat((all_padding_mask, padding_mask.unsqueeze(0)), dim=0)
        all_esm_vec = all_esm_vec[1:, :, :]
        all_label = all_label[1:, :]
        all_mask = all_mask[1:, :]
        all_padding_mask = all_padding_mask[1:, :]
        dataset = TensorDataset(all_esm_vec, all_label, all_mask, all_padding_mask)
        torch.save(dataset, "your/path/to/dataset/")


if __name__ == '__main__':
    # ----------------------------------------------------------------------------------------------------
    '''this program is used to create a tensordataset based on the esm-2'''
    from utils import extract_protein_sequence, refdb_find_shift, refdb_get_cs_seq, refdb_get_shift_re
    from utils import align_bmrb_pdb
    import os
    atom_types = ["CA","CB","C","N","H","HA"]
    refdb = "\dateset\RefDB_test_remove"
    save_path = "your/path/to/dataset/"
    for atom_type in atom_types:
        main(refdb, save_path, atom_type)


