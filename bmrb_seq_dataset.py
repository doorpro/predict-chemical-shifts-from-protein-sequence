#!/usr/bin/env python
# -*- coding = utf-8 -*-
# @Time : 2023/7/31 20:53
# @Author : door
# @File : bmrb_seq_dataset.py
# @Software : PyCharm
# @File : bmrb_seq_dataset.py
# @desc:
'''该文件用于创建仅基于bmrb数据库中信息的dataset，因为基于序列的蛋白质化学位移预测仅仅需要序列信息故不需要高质量的pdb-bmrb数据集。
而pdb文件中常常与bmrb中的序列难以对应'''
import torch
from pynmrstar import Entry
import pickle

def extract_protein_sequence(entry):
    '''该程序从nmrstar文件中的entity_poly_seq中读取该蛋白质的序列'''
    nmrstar_file = 'F:\\nmrprediction\dataset\\all_bmrb\\' + str(entry) + '.str'
    entry = Entry.from_file(nmrstar_file)
    amino_list = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','U']
    amino_list2 = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','U']
    # 此处加入了不常见的U
    def amino2AA(amino):
        for i in range(21):
            if amino == amino_list[i]:
                AA = amino_list2[i]
                return AA
            elif amino == amino_list2[i]:
                return amino
        return '_'
    # 若存在不包含在列表内的氨基酸类型则返回_
    sequence_data = entry.get_loops_by_category('entity_poly_seq')
    bmrb_seq_list = ['' for i in range(len(sequence_data))]
    entity = -1
    # 表示当前在迭代第几个entity
    for i in sequence_data:
        entity += 1
        bmrb_seq = ''
        for row in i:
            AA = amino2AA(row[1])
            bmrb_seq += AA
        bmrb_seq_list[entity] = bmrb_seq
    # 提取序列信息中的氨基酸代码

    return bmrb_seq_list

def get_shifts(entry, atom_type, bmrb_seq_list):
    '''输入bmrbid, 原子类型（例如CA或HA），以及bmrb中包含所有entity的序列的列表
    输出的shift_list_all为所有entity的化学位移的列表， mask_all为所有entity的掩码'''
    entity_num = len(bmrb_seq_list)
    shift_list_all = [0 for i in range(entity_num)]
    mask_all = [0 for i in range(entity_num)]
    entry = Entry.from_file(
        'F:\\nmrprediction\dataset\\all_bmrb\\' + str(entry) + '.str')
    chem_shifts = entry.get_loops_by_category('Atom_chem_shift')
    for entity in range(entity_num):
        # entity表示正在处理第几个entity
        shift_list = [0 for i in range(len(bmrb_seq_list[entity]))]
        mask = [False for i in range(len(bmrb_seq_list[entity]))]
        for chem_shift in chem_shifts:
            for i in range(len(chem_shift['ID'])):
                if atom_type == chem_shift['Atom_ID'][i] and int(chem_shift['Entity_ID'][i]) == entity + 1:
                    shift_list[int(chem_shift['Seq_ID'][i]) - 1] = float(chem_shift['Val'][i])
                    mask[int(chem_shift['Seq_ID'][i]) - 1] = True
        shift_list_all[entity] = shift_list
        mask_all[entity] = mask
    return shift_list_all, mask_all

def get_HA_shifts(entry, atom_type, bmrb_seq_list):
    '''与get_shifts相似功能，但是由于HA的特殊性，即GLY包含HA2和HA3，故重新写一个函数，该函数以HA3为标签值'''
    entity_num = len(bmrb_seq_list)
    shift_list_all = [0 for i in range(entity_num)]
    mask_all = [0 for i in range(entity_num)]
    entry = Entry.from_file(
        'F:\\nmrprediction\dataset\\all_bmrb\\' + str(entry) + '.str')
    chem_shifts = entry.get_loops_by_category('Atom_chem_shift')
    for entity in range(entity_num):
        # entity表示正在处理第几个entity
        shift_list = [0 for i in range(len(bmrb_seq_list[entity]))]
        mask = [False for i in range(len(bmrb_seq_list[entity]))]
        for chem_shift in chem_shifts:
            for row in chem_shift:
                # if row[7] == atom_type or row[7] == 'HA2' and int(row[3]) == entity + 1:
                if atom_type in row[7] and int(row[3]) == entity + 1:
                    shift_list[int(row[5]) - 1] = float(row[10])
                    mask[int(row[5]) - 1] = True
        shift_list_all[entity] = shift_list
        mask_all[entity] = mask
    return shift_list_all, mask_all

def nmr_len(bmrbid, row_len):
    entry = Entry.from_file(
        'F:\\nmrprediction\\dataset\\all_bmrb\\' + str(bmrbid) + '.str')
    chem_shifts = entry.get_loops_by_category('Atom_chem_shift')
    for chem_shift in chem_shifts:
        for row in chem_shift:
            if len(row) == row_len:
                return True
            else:
                return False


def extract_from_star2(file):
    amino_list2 = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V',
                   'U']
    start_line = 99999
    bmrb_seq = ''
    with open(file, encoding="utf-8") as entry:
        for i, line in enumerate(entry):
            if "_Mol_residue_sequence" in line:
                start_line = i + 1
            if line.strip() == ";" and start_line < i < start_line + 50:
                end_line = i
                break

    with open(file, encoding="utf-8") as entry:
        for i, line in enumerate(entry):
            if start_line < i < end_line:
                for aa in line.strip():
                    if aa in amino_list2:
                        bmrb_seq += aa
                    else:
                        bmrb_seq += '_'
    bmrb_seq_list = []
    bmrb_seq_list.append(bmrb_seq)
    return bmrb_seq

def refdb_get_seq(file):
    amino_list2 = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','U']
    start_line = 99999
    bmrb_seq = ''
    with open(file, encoding="utf-8") as entry:
        for i, line in enumerate(entry):
            if "_Mol_residue_sequence" in line:
                start_line = i + 1
            if line.strip() == ";" and start_line < i < start_line + 50:
                end_line = i
                break

    with open(file, encoding="utf-8") as entry:
        for i, line in enumerate(entry):
            if start_line < i < end_line:
                for aa in line.strip():
                    if aa in amino_list2:
                        bmrb_seq+=aa
                    else:
                        bmrb_seq += '_'
    return bmrb_seq

def refdb_find_shift(file):
    start_line = 99999
    with open(file, encoding='utf-8') as entry:
        for i, line in enumerate(entry):
            if "_Chem_shift_ambiguity_code" in line:
                start_line = i + 1
            if i > start_line and "stop_" in line:
                end_line = i - 1
                break
    return start_line, end_line

def refdb_get_shift(file, s, e, bmrb_seq):
    res = len(bmrb_seq)
    shift = torch.zeros(res)
    mask = torch.zeros(res).bool()
    with open(file, encoding='utf-8') as entry:
        for i, line in enumerate(entry):
            if s < i < e:
                res_id = line.split()[2]
                cs = line.split()[6]
                atom = line.split()[4]
                if atom == "CA":
                    shift[int(res_id) - 1] = float(cs)
                    mask[int(res_id) - 1] = True
    return shift, mask

def aa(amino):
    amino_list = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',
                  'PRO',
                  'SER', 'THR', 'TRP', 'TYR', 'VAL', 'U']
    amino_list2 = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y',
                   'V', 'U']
    for i in range(21):
        if amino == amino_list[i]:
            return amino_list2[i]
    return "_"

def refdb_get_cs_seq(file_path, s, e):
    '''从nmrstar文件中读出cs表对应的序列'''
    cs_seq = ""
    start_resid = 99999
    res_num = -1
    with open(file_path, encoding='utf-8') as entry:
        for i, line in enumerate(entry):
            if s < i < e:
                res = line.split()[2]
                res_id = int(line.split()[1])
                res = aa(res)
                if res_id != start_resid:
                    cs_seq += res
                    start_resid = res_id
                    res_num += 1
    return cs_seq

def refdb_get_shift_re(file, s, e, bmrb_seq, matched):
    '''输入nmrstar文件，以及其中的CS表对应的行数，fasta序列和对其的列表matched，输出长度与输入序列相同的列表，分别是shifts和mask'''
    res = len(bmrb_seq)
    shift = torch.zeros(res)
    mask = torch.zeros(res).bool()
    res_num = -1
    # 表示当前迭代到cs表中第几个氨基酸
    start_res_num = 99999
    with open(file, encoding='utf-8') as entry:
        for i, line in enumerate(entry):
            if s < i < e:
                res_id = int(line.split()[1])
                if res_id != start_res_num:
                    start_res_num = res_id
                    res_num += 1
                cs = line.split()[5]
                atom = line.split()[3]
                if atom == "N" and res_num < len(matched):
                # if "HA" in atom and res_num < len(matched):
                    # resnum必须小于seq的长度，多出不要
                    if type(matched[res_num]) == type(9):
                        # 使得对应的必须为数字而不是match列表中的False值
                        shift[int(matched[res_num])] = float(cs)
                        mask[int(matched[res_num])] = True
    return shift, mask

if __name__ == '__main__':
    # del_list = [4356, 7197]
    # with open("F:\\nmrprediction\\CSpred-master\\CSpred-master\\train_model\\pdb_bmr_dict.pkl",
    #           "rb") as f:
    #     pdb_bmrb_dict = pickle.load(f)
    #     for bmrbid in pdb_bmrb_dict.values():
    #         if str(bmrbid).isdigit() and bmrbid not in del_list:
    #             print(bmrbid)
    #             bmrb_seq_list = extract_protein_sequence(bmrbid)
    #             shift_list, mask_list = get_shifts(bmrbid, 'CA', bmrb_seq_list)

    # import os
    # all_bmrb = 'F:\\nmrprediction\\dataset\\separ_bmrb\\8'
    # for root, directories, files in os.walk(all_bmrb):
    #     for file in files:
    #         bmrbid = file.split('.')[0]
    #         a = nmr_len(bmrbid)
    #         if a == 25:
    #             print(bmrbid)
    #             print(a)



    # bmrb_seq_list = extract_protein_sequence(19424)
    # shift_list, mask_list = get_shifts(19424, 'CA', bmrb_seq_list)
    # print(shift_list)
    bmrb_seq_list = extract_protein_sequence(7141)
    print(bmrb_seq_list)
    shift_list, mask_list = get_shifts(7141, 'CA', bmrb_seq_list)
    print(shift_list)

