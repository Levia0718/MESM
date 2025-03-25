import nni
import time
import argparse
import json
import os
import torch
import warnings
import sys

current_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# prevent ModuleNotFoundError:No module named‘XXX‘
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


def train_ppi(model, graph, loss_func, optimizer, param, scheduler, log_file):
    # test
    global_best_test_f1 = 0.0
    global_best_test_f1_epoch = 0

    for epoch in range(param['epochs']):
        train_loss, train_f1 = run_one_epoch(model, graph, loss_func, optimizer, scheduler, mode='train')

        if (epoch + 1) % 1 == 0:
            test_loss, test_f1 = run_one_epoch(model, graph, loss_func, optimizer, scheduler, mode='test')

            if global_best_test_f1 < test_f1:
                global_best_test_f1 = test_f1
                global_best_test_f1_epoch = epoch + 1

                torch.save(
                    model.state_dict(), os.path.join(param['save_path'], 'ppi_test_best.pth')
                )

            print(
                "Epoch [{}/{}] | \033[91mTrain: loss: {:.5f}, F1: {:.5f}\033[0m "
                " | \033[94mTest: loss: {:.5f}, F1: {:.5f}\033[0m "
                " | \033[96mBest test_f1: {:.5f}, in {} epoch\033[0m"
                .format(epoch+1, param["epochs"], train_loss, train_f1, test_loss, test_f1, global_best_test_f1, global_best_test_f1_epoch)
            )
            log_file.write(
                "Epoch [{}/{}] | Train: loss: {:.5f}, F1: {:.5f}"
                " | Test: loss: {:.5f}, F1: {:.5f}"
                " | Best test_f1: {:.5f}, in {} epoch\n"
                .format(epoch+1, param["epochs"], train_loss, train_f1, test_loss, test_f1, global_best_test_f1, global_best_test_f1_epoch)
            )
            log_file.flush()
        else:
            print(
                "Epoch [{}/{}] | \033[91mTrain: loss: {:.5f}, F1: {:.5f}\033[0m"
                .format(epoch+1, param["epochs"], train_loss, train_f1)
            )
            log_file.write(
                "Epoch [{}/{}] | Train: loss: {:.5f}, F1: {:.5f}\n"
                .format(epoch+1, param["epochs"], train_loss, train_f1)
            )
            log_file.flush()
        torch.cuda.empty_cache()
    log_file.close()

    os.rename(param['save_path'], '{}/results/{}/{}/MESM_{:.5f}'.format(current_file_path, param['dataset'], param['split_mode'], global_best_test_f1))


def train_ppi_pre(param):
    param = set_param(param)

    if not os.path.exists(param['save_path']):
        os.makedirs(param['save_path'])

    log_file = open(os.path.join(param['save_path'], "train_log.txt"), 'a+')

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    scheduler = get_stepwise_decay_schedule_with_warmup(optimizer, 0, 10000, 0.95)
    loss_func = ZLPR_loss(mode='soft_label').to(device)

    graph = graph.to(device)

    train_ppi(
        model, graph, loss_func, optimizer, param, scheduler, log_file
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="SHS27k")
    parser.add_argument("--split_mode", type=str, default="bfs")

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

    train_ppi_pre(param)
