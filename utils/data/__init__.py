import os
import pickle
import numpy as np
import torch
import json
import random
from torch_geometric.data import Data
from tqdm import tqdm

from utils.normal import get_bfs_sub_graph, get_dfs_sub_graph


current_file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GNN_DATA:
    def __init__(self, protein_actions_path, skip_head=True, graph_undirection=True):
        self.protein_name_dict = {}
        self.ppi_dict = {}
        self.ppi_label_list = []
        self.ppi_list = []
        self.seven_ppi_list = [[] for _ in range(7)]
        self.pretrained_protein_dict = {}
        self.protein_dict = {}
        self.x = None

        self.edge_index = None
        self.seven_edge_index = []
        self.edge_label = None
        self.data = None
        self.ppi_split_dict = None

        protein_index = 0
        ppi_index = 0
        class_map = {'reaction': 0, 'binding': 1, 'ptmod': 2, 'activation': 3, 'inhibition': 4, 'catalysis': 5, 'expression': 6}

        for line in tqdm(open(protein_actions_path)):
            if skip_head:
                skip_head = False
                continue

            column = line.strip().split('\t')
            if column[0] not in self.protein_name_dict.keys():
                self.protein_name_dict[column[0]] = protein_index
                protein_index += 1
            if column[1] not in self.protein_name_dict.keys():
                self.protein_name_dict[column[1]] = protein_index
                protein_index += 1

            if column[0] < column[1]:
                temp_data = column[0] + "__" + column[1]
            else:
                temp_data = column[1] + "__" + column[0]

            temp_seven = temp_data.strip().split('__')
            if temp_data not in self.ppi_dict.keys():
                self.ppi_dict[temp_data] = ppi_index
                ppi_index += 1

                temp_label = [0, 0, 0, 0, 0, 0, 0]
                temp_label[class_map[column[2]]] = 1
                self.ppi_label_list.append(temp_label)

                self.seven_ppi_list[class_map[column[2]]].append(temp_seven)
            else:
                index = self.ppi_dict[temp_data]
                temp_label = self.ppi_label_list[index]
                if temp_label[class_map[column[2]]] != 1:
                    temp_label[class_map[column[2]]] = 1
                    self.ppi_label_list[index] = temp_label

                    self.seven_ppi_list[class_map[column[2]]].append(temp_seven)

        for ppi in tqdm(self.ppi_dict.keys()):
            temp = ppi.strip().split('__')
            self.ppi_list.append(temp)
        ppi_num = len(self.ppi_list)
        assert ppi_num == len(self.ppi_label_list)

        for i in tqdm(range(ppi_num)):
            pro1_name = self.ppi_list[i][0]
            pro2_name = self.ppi_list[i][1]
            self.ppi_list[i][0] = self.protein_name_dict[pro1_name]
            self.ppi_list[i][1] = self.protein_name_dict[pro2_name]
        for i in range(7):
            for j in tqdm(range(len(self.seven_ppi_list[i]))):
                pro1_name = self.seven_ppi_list[i][j][0]
                pro2_name = self.seven_ppi_list[i][j][1]
                self.seven_ppi_list[i][j][0] = self.protein_name_dict[pro1_name]
                self.seven_ppi_list[i][j][1] = self.protein_name_dict[pro2_name]

        if graph_undirection:
            for i in tqdm(range(ppi_num)):
                temp_ppi = self.ppi_list[i][::-1]
                temp_ppi_label = self.ppi_label_list[i]
                self.ppi_list.append(temp_ppi)
                self.ppi_label_list.append(temp_ppi_label)

            for i in range(7):
                for j in tqdm(range(len(self.seven_ppi_list[i]))):
                    temp_ppi = self.seven_ppi_list[i][j][::-1]
                    self.seven_ppi_list[i].append(temp_ppi)

        self.node_num = len(self.protein_name_dict)
        self.edge_num = len(self.ppi_list)

    def get_feature_pretrained(self):
        pretrained_protein_path = os.path.join(current_file_path, 'datasets/protein_data/multimodal_protein_representations/all_protein_STRING.pickle')
        # pretrained_protein_path = os.path.join(current_file_path, 'datasets/protein_data/multimodal_protein_representations/all_protein_Yeast.pickle')
        with open(pretrained_protein_path, 'rb') as f:
            self.pretrained_protein_dict = pickle.load(f)
        f.close()

        for name in tqdm(self.protein_name_dict.keys()):
            self.protein_dict[name] = np.array(
                self.pretrained_protein_dict[name]
            )

        print('Protein number: ', len(self.protein_dict))

    def generate_data(self):
        ppi_list = np.array(self.ppi_list)
        self.edge_index = torch.tensor(ppi_list, dtype=torch.long)

        for i in range(7):
            current_ppi_list = np.array(self.seven_ppi_list[i])
            current_edge_index = torch.tensor(current_ppi_list, dtype=torch.long).T
            self.seven_edge_index.append(current_edge_index)

        ppi_label_list = np.array(self.ppi_label_list)
        self.edge_label = torch.tensor(ppi_label_list, dtype=torch.float32)

        self.x = []
        for name in self.protein_name_dict:
            self.x.append(self.protein_dict[name])
        self.x = np.array(self.x)
        self.x = torch.tensor(self.x, dtype=torch.float)

        self.data = Data(x=self.x, edge_index=self.edge_index.T, seven_edge_index=self.seven_edge_index, edge_label=self.edge_label)

    def split_dataset_train_test(self, train_test_index_path, test_size=0.2, mode='random'):
        if not os.path.exists(train_test_index_path):
            if mode == 'random':
                ppi_num = int(self.edge_num // 2)
                random_list = [i for i in range(ppi_num)]
                random.shuffle(random_list)

                self.ppi_split_dict = {
                    'train_index': random_list[: int(ppi_num * (1 - test_size))],
                    'test_index': random_list[int(ppi_num * (1 - test_size)):]
                }

                jsobj = json.dumps(self.ppi_split_dict)
                with open(train_test_index_path, 'w') as f:
                    f.write(jsobj)
                    f.close()
            elif mode == 'bfs' or mode == 'dfs':
                print("use {} methed split train and valid dataset".format(mode))
                node_to_edge_index = {}
                edge_num = int(self.edge_num // 2)
                for i in range(edge_num):
                    edge = self.ppi_list[i]
                    if edge[0] not in node_to_edge_index.keys():
                        node_to_edge_index[edge[0]] = []
                    node_to_edge_index[edge[0]].append(i)

                    if edge[1] not in node_to_edge_index.keys():
                        node_to_edge_index[edge[1]] = []
                    node_to_edge_index[edge[1]].append(i)

                node_num = len(node_to_edge_index)

                sub_graph_size = int(edge_num * test_size)
                if mode == 'bfs':
                    selected_edge_index = get_bfs_sub_graph(self.ppi_list, node_num, node_to_edge_index, sub_graph_size)
                elif mode == 'dfs':
                    selected_edge_index = get_dfs_sub_graph(self.ppi_list, node_num, node_to_edge_index, sub_graph_size)

                all_edge_index = [i for i in range(edge_num)]

                unselected_edge_index = list(set(all_edge_index).difference(set(selected_edge_index)))

                self.ppi_split_dict = {
                    'train_index': unselected_edge_index,
                    'test_index': selected_edge_index
                }

                assert len(unselected_edge_index) + len(selected_edge_index) == edge_num

                jsobj = json.dumps(self.ppi_split_dict)
                with open(train_test_index_path, 'w') as f:
                    f.write(jsobj)
                    f.close()

            else:
                print("your mode is {}, you should use bfs, dfs or random".format(mode))
                return
        else:
            with open(train_test_index_path, 'r') as f:
                self.ppi_split_dict = json.load(f)
                f.close()
