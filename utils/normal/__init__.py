import torch
import numpy as np
import random
from torch.optim.lr_scheduler import LambdaLR
import os

current_file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()

        if num_devices > 1:
            device = torch.device('cuda:1')
        else:
            device = torch.device('cuda:0')
    else:
        raise RuntimeError("No CUDA devices available. Please check your installation or hardware.")

    return device


# Data splitting by BFS
def get_bfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size):
    candiate_node = []
    selected_edge_index = []
    selected_node = []

    random_node = random.randint(0, node_num - 1)
    while len(node_to_edge_index[random_node]) > 5:
        random_node = random.randint(0, node_num - 1)
    candiate_node.append(random_node)

    while len(selected_edge_index) < sub_graph_size:
        cur_node = candiate_node.pop(0)
        selected_node.append(cur_node)
        for edge_index in node_to_edge_index[cur_node]:

            if edge_index not in selected_edge_index:
                selected_edge_index.append(edge_index)

                end_node = -1
                if ppi_list[edge_index][0] == cur_node:
                    end_node = ppi_list[edge_index][1]
                else:
                    end_node = ppi_list[edge_index][0]

                if end_node not in selected_node and end_node not in candiate_node:
                    candiate_node.append(end_node)
            else:
                continue

    # node_list = candiate_node + selected_node

    return selected_edge_index


# Data splitting by DFS
def get_dfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size):
    stack = []
    selected_edge_index = []
    selected_node = []

    random_node = random.randint(0, node_num - 1)
    while len(node_to_edge_index[random_node]) > 5:
        random_node = random.randint(0, node_num - 1)
    stack.append(random_node)

    while len(selected_edge_index) < sub_graph_size:
        # print(len(selected_edge_index), len(stack), len(selected_node))
        cur_node = stack[-1]
        if cur_node in selected_node:
            flag = True
            for edge_index in node_to_edge_index[cur_node]:
                if flag:
                    end_node = -1
                    if ppi_list[edge_index][0] == cur_node:
                        end_node = ppi_list[edge_index][1]
                    else:
                        end_node = ppi_list[edge_index][0]

                    if end_node in selected_node:
                        continue
                    else:
                        stack.append(end_node)
                        flag = False
                else:
                    break
            if flag:
                stack.pop()
            continue
        else:
            selected_node.append(cur_node)
            for edge_index in node_to_edge_index[cur_node]:
                if edge_index not in selected_edge_index:
                    selected_edge_index.append(edge_index)

    return selected_edge_index


class Metrictor_PPI:
    def __init__(self, pre_y, truth_y, is_binary=False):
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0

        if is_binary:
            length = pre_y.shape[0]
            for i in range(length):
                if pre_y[i] == truth_y[i]:
                    if truth_y[i] == 1:
                        self.TP += 1
                    else:
                        self.TN += 1
                elif truth_y[i] == 1:
                    self.FN += 1
                elif pre_y[i] == 1:
                    self.FP += 1
            self.num = length

        else:
            N, C = pre_y.shape
            for i in range(N):
                for j in range(C):
                    if pre_y[i][j] == truth_y[i][j]:
                        if truth_y[i][j] == 1:
                            self.TP += 1
                        else:
                            self.TN += 1
                    elif truth_y[i][j] == 1:
                        self.FN += 1
                    elif truth_y[i][j] == 0:
                        self.FP += 1
            self.num = N * C

    def show_result(self, is_print=False):
        self.Accuracy = (self.TP + self.TN) / (self.num + 1e-10)
        self.Precision = self.TP / (self.TP + self.FP + 1e-10)
        self.Recall = self.TP / (self.TP + self.FN + 1e-10)
        self.F1 = 2 * self.Precision * self.Recall / (self.Precision + self.Recall + 1e-10)
        if is_print:
            print("Accuracy: {}".format(self.Accuracy))
            print("Precision: {}".format(self.Precision))
            print("Recall: {}".format(self.Recall))
            print("F1-Score: {}".format(self.F1))


def get_stepwise_decay_schedule_with_warmup(optimizer, num_warmup_steps, num_steps_decay, decay_rate, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        num_fades = (current_step-num_warmup_steps)//num_steps_decay
        return (decay_rate**num_fades)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def set_param(param):
    param["protein_actions_path"] = "{}/datasets/processed_data_{}/protein.actions.{}.txt".format(
        current_file_path, param["dataset"], param["dataset"]
    )
    param["train_test_index_path"] = "{}/datasets/processed_data_{}/split/train_test/{}.json".format(
        current_file_path, param['dataset'], param['split_mode']
    )
    param["save_path"] = "{}/results/{}/{}/MESM_{}".format(
        current_file_path, param['dataset'], param['split_mode'], param["timestamp"]
    )

    return param
