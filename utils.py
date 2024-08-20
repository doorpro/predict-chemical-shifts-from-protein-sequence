'''
coding:utf-8
@software:
@Time:2024/7/30 22:05
@Author:zhuhe
'''
import torch
from pynmrstar import Entry
import copy

def extract_protein_sequence(entry):
    '''该函数从nmrstar文件中的entity_poly_seq中读取该蛋白质的序列
    read protein sequences from nmrstar'''
    nmrstar_file = 'F:\\nmrprediction\\CSpre\\dataset\\all_bmrb\\' + str(entry) + '.str'
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
        return 'X'
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

def refdb_get_seq(file):
    amino_list2 = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V'] # "U" is not included
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
                        bmrb_seq += 'X'
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
                res_id = line.split()[1]
                cs = line.split()[5]
                atom = line.split()[3]
                if atom == "HA":
                    shift[int(res_id) - 1] = float(cs)
                    mask[int(res_id) - 1] = True
    return shift, mask

def align_bmrb_pdb(bmrb_seq, pdb_seq):
    '''输入bmrb和pdb的序列，输出一个列表，该列表长度与pdb文件的序列相同，每个对应位置储存的是该res在bmrb序列中的位置'''
    matched_seq = [False for j in range(len(pdb_seq))]
    # 长度与pdb文件的res数相同，相应位置表示
    findmatch = 0
    index_bmrb = 0
    copy_index = 0
    start_index = 0
    # 用来表示第一次匹配是否完成
    index_pdb = 0
    # 用于储存目前匹配到bmrb第几个res
    for i in range(len(pdb_seq)):

        # if findmatch == 1:
        # 如果按上一行，则有可能pdbseq和bmrbseq对齐，但是pdbseq多出一位，则会出现index_bmrb比bmrb_seq多出一位的情况而报错
        if findmatch == 1 and index_bmrb < len(bmrb_seq):
            if bmrb_seq[index_bmrb] == pdb_seq[i]:
                matched_seq[i] = index_bmrb
                index_bmrb += 1
            else:
                findmatch = 0

        if findmatch == 0 and i < len(pdb_seq)-2:
            # 表示此时bmrb和pdb的序列不匹配需要重新查找匹配点
            if start_index == 0:
                copy_index = copy.copy(index_bmrb)
                # 在设置循环的时候用copyindex替代indexbmrb防止bug
                for query_bmrb in range(copy_index, len(bmrb_seq)-2 if len(bmrb_seq)-2 < copy_index+30 else copy_index+30):
                # for query_bmrb in range(len(bmrb_seq) - 2):
                    # print(i, query_bmrb)
                    if bmrb_seq[query_bmrb] == pdb_seq[i] and bmrb_seq[query_bmrb + 1] == pdb_seq[i + 1]\
                            and bmrb_seq[query_bmrb + 2] == pdb_seq[i + 2]:
                        findmatch = 1
                        index_bmrb = query_bmrb
                        start_index = 1
                        matched_seq[i] = index_bmrb
                        index_bmrb += 1
                        break
            if start_index == 1:
                copy_index = copy.copy(index_bmrb)
                # for query_bmrb in range(copy_index, len(bmrb_seq)-2):
                for query_bmrb in range(copy_index, len(bmrb_seq) - 2 if len(bmrb_seq)-2 < copy_index+10 else copy_index+10):
                # for query_bmrb in range(len(bmrb_seq) - 2):
                    if bmrb_seq[query_bmrb] == pdb_seq[i] and bmrb_seq[query_bmrb + 1] == pdb_seq[i + 1]\
                            and bmrb_seq[query_bmrb + 2] == pdb_seq[i + 2]:
                        findmatch = 1
                        index_bmrb = query_bmrb
                        start_index = 1
                        matched_seq[i] = index_bmrb
                        index_bmrb += 1
                        break
    return matched_seq
def aa(amino):
    amino_list = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',
                  'PRO',
                  'SER', 'THR', 'TRP', 'TYR', 'VAL']
    amino_list2 = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y',
                   'V']
    for i in range(20):
        if amino == amino_list[i]:
            return amino_list2[i]
    return "<mask>"
def refdb_get_cs_seq(file_path, s, e):
    '''从nmrstar文件中读出cs表对应的序列'''
    cs_seq = ""
    start_resid = 99999
    res_num = -1
    with open(file_path, encoding='utf-8') as entry:
        for i, line in enumerate(entry):
            if s < i < e:
                res = line.split()[3]
                res_id = int(line.split()[2])
                res = aa(res)
                if res_id != start_resid:
                    cs_seq += res
                    start_resid = res_id
                    res_num += 1
    return cs_seq

def refdb_get_shift_re(file, s, e, bmrb_seq, matched, atom_type):
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
                res_id = int(line.split()[2])
                if res_id != start_res_num:
                    start_res_num = res_id
                    res_num += 1
                cs = line.split()[6]
                atom = line.split()[4]
                if atom_type == "HA":
                    if "HA" in atom and res_num < len(matched):
                        # resnum必须小于seq的长度，多出不要
                        if type(matched[res_num]) == type(9):
                            # 使得对应的必须为数字而不是match列表中的False值
                            shift[int(matched[res_num])] = float(cs)
                            mask[int(matched[res_num])] = True
                else:
                    if atom == atom_type and res_num < len(matched):
                    # if "HA" in atom and res_num < len(matched):
                        # resnum必须小于seq的长度，多出不要
                        if type(matched[res_num]) == type(9):
                            # 使得对应的必须为数字而不是match列表中的False值
                            shift[int(matched[res_num])] = float(cs)
                            mask[int(matched[res_num])] = True
    return shift, mask

def get_shifts(entry, atom_type, bmrb_seq_list):
    '''输入bmrbid, 原子类型（例如CA或HA），以及bmrb中包含所有entity的序列的列表
    输出的shift_list_all为所有entity的化学位移的列表， mask_all为所有entity的掩码'''
    entity_num = len(bmrb_seq_list)
    shift_list_all = [0 for i in range(entity_num)]
    mask_all = [0 for i in range(entity_num)]
    entry = Entry.from_file(
        'F:\\nmrprediction\\CSpre\\dataset\\all_bmrb\\' + str(entry) + '.str')
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
        'F:\\nmrprediction\\CSpre\\dataset\\all_bmrb\\' + str(entry) + '.str')
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

if __name__ == "__main__":
    print(True)