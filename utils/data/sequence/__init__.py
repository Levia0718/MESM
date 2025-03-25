import os
import pickle
import numpy as np
import torch
from tqdm import tqdm

current_file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def get_protein_aac(pseq_path):
    pseq_path = pseq_path
    pseq_list = []

    for line in tqdm(open(pseq_path)):
        line = line.strip().split('\t')
        if line[0] not in pseq_list:
            pseq_list.append(line[1])

    print("protein num: {}".format(len(pseq_list)))

    return pseq_list


def embed_normal(seq, dim, max_len=2000):
    if len(seq) > max_len:
        return seq[:max_len]
    elif len(seq) < max_len:
        less_len = max_len - len(seq)
        return np.concatenate((seq, np.zeros((less_len, dim))))
    return seq


def vectorize(vec_path, pseq_list):
    acid2vec = {}
    dim = None
    for line in open(vec_path):
        line = line.strip().split('\t')
        temp = np.array([float(x) for x in line[1].split()])
        acid2vec[line[0]] = temp
        if dim is None:
            dim = len(temp)

    pvec_list = []

    for p_seq in tqdm(pseq_list):
        temp_seq = p_seq
        temp_vec = []
        for acid in temp_seq:
            temp_vec.append(acid2vec[acid])
        temp_vec = np.array(temp_vec)

        temp_vec = embed_normal(temp_vec, dim)

        pvec_list.append(temp_vec)

    return pvec_list


def get_feature_origin(pseq_path, vec_path):
    pseq_list = get_protein_aac(pseq_path)
    pvec_list = vectorize(vec_path, pseq_list)

    print('Protein number: ', len(pvec_list))

    return pvec_list


if __name__ == "__main__":
    protein_vec_list = get_feature_origin('{}/datasets/raw_data/STRING_sequence.tsv'.format(current_file_path), '{}/datasets/raw_data/vec5_CTC.txt'.format(current_file_path))
    # protein_vec_list = get_feature_origin('{}/datasets/raw_data/Yeast_sequence.tsv'.format(current_file_path), '{}/datasets/raw_data/vec5_CTC.txt'.format(current_file_path))

    x = []
    for vec in protein_vec_list:
        x.append(vec)
    x = np.array(x)
    x = torch.tensor(x, dtype=torch.float)

    save_path = "{}/datasets/protein_data".format(current_file_path)
    with open(f'{save_path}/sequences_STRING.pkl', 'wb') as f:
        pickle.dump(x, f)
    # with open(f'{save_path}/sequences_Yeast.pkl', 'wb') as f:
    #     pickle.dump(x, f)

    print("Done")
