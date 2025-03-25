import nni
import time
import argparse
import json
import os
import torch
import warnings
import sys

# 项目的根目录
current_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 命令行运行.py文件时，需要手动添加路径检索，才不会出现ModuleNotFoundError:No module named‘XXX‘
sys.path.append(current_file_path)

from utils.normal import set_seed, get_stepwise_decay_schedule_with_warmup, set_param, get_device
from utils.normal.one_epoch import run_one_epoch
from utils.loss import ZLPR_loss
from utils.subgraph import transform_subgraphs
from utils.data import GNN_DATA
from utils.se import add_random_walk_se
from model.ppi_prediction_model import MESM

warnings.filterwarnings("ignore", category=Warning)

try:
    device = get_device()
    print(f'Selected device: {device}')
except RuntimeError as e:
    print(e)


def test_ppi(model, graph, loss_func, optimizer, param, scheduler):
    log_file = open(os.path.join(param['save_path'], "train_log.txt"), 'a+')

    test_loss, test_f1 = run_one_epoch(
        model, graph, loss_func, optimizer, scheduler, mode='test'
    )

    print(
        "\033[94mTest: loss: {:.5f}, F1: {:.5f}\033[0m ".format(test_loss, test_f1)
    )
    log_file.write(
        "Test: loss: {:.5f}, F1: {:.5f}".format(test_loss, test_f1)
    )
    log_file.flush()
    log_file.close()


def test_ppi_pre(param):
    param = set_param(param)

    ppi_data = GNN_DATA(param['protein_actions_path'])
    ppi_data.get_feature_pretrained()
    ppi_data.generate_data()
    graph = ppi_data.data

    ppi_data.split_dataset_train_test(param['train_test_index_path'], mode=param['split_mode'])
    graph.train_mask = ppi_data.ppi_split_dict['train_index']
    graph.test_mask = ppi_data.ppi_split_dict['test_index']
    print(
        "train ppi nums: {}, test ppi nums: {}, total ppi nums: {}".format(
            len(graph.train_mask), len(graph.test_mask),
            len(graph.train_mask) + len(graph.test_mask)
        )
    )

    graph = transform_subgraphs(graph)
    se_dim = 20
    add_random_walk_se(graph, walk_length=se_dim)
    model = MESM(hidden=1024, class_num=7, se_dim=se_dim, param=param).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)  # RoseTTAFold2-PPI
    scheduler = get_stepwise_decay_schedule_with_warmup(optimizer, 0, 10000, 0.95)  # RoseTTAFold2-PPI
    loss_func = ZLPR_loss(mode='soft_label').to(device)

    if not os.path.exists(param['save_path']):
        os.makedirs(param['save_path'])

    load_path = "{}/results/{}/{}/{}".format(
            current_file_path, args.dataset, param['split_mode'], args.model_save_path
        )
    model.load_state_dict(torch.load(os.path.join(load_path, 'ppi_test_best.pth')))

    graph = graph.to(device)

    test_ppi(
        model, graph, loss_func, optimizer, param, scheduler
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="SHS27k")
    parser.add_argument("--split_mode", type=str, default="bfs")
    parser.add_argument("--model_save_path", type=str, default=None)

    args = parser.parse_args()
    param = args.__dict__
    param.update(nni.get_next_parameter())

    timestamp = time.strftime("%Y-%m-%d %H-%M-%S") + f"-%3d" % ((time.time() - int(time.time())) * 1000)
    param["timestamp"] = timestamp

    new_param = None
    if os.path.exists("{}/config/params.json".format(current_file_path)):
        new_param = json.loads(
            open("{}/config/params.json".format(current_file_path), 'r').read()
        )[param['dataset']][param['split_mode']]
    param.update(new_param)

    set_seed(param['seed'])

    test_ppi_pre(param)
