import math
import torch.nn as nn
import torch
import os

from utils.normal import Metrictor_PPI, get_device

try:
    device = get_device()
    print(f'Selected device: {device}')
except RuntimeError as e:
    print(e)

current_file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_one_epoch(model, graph, loss_func, optimizer, scheduler, mode='train'):
    f1_sum = 0.0
    loss_sum = 0.0
    steps = None

    accuracy_sum = 0.0
    precision_sum = 0.0
    recall_sum = 0.0

    if mode == 'train':
        model.train()
        batch_size = len(graph.train_mask)
        steps = math.ceil(len(graph.train_mask) / batch_size)

        for step in range(steps):
            if step == steps - 1:
                edge_id = graph.train_mask[step * batch_size:]
            else:
                edge_id = graph.train_mask[step * batch_size: step * batch_size + batch_size]

            output = model(graph, edge_id)

            label = graph.edge_label[edge_id].to(device)
            loss_all = loss_func(output, label)

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            m = nn.Sigmoid()
            output = (m(output) > 0.5).type(torch.FloatTensor)
            metrics = Metrictor_PPI(output.cpu().data, label.cpu().data)
            metrics.show_result()

            f1_sum += metrics.F1
            accuracy_sum += metrics.Accuracy
            precision_sum += metrics.Precision
            recall_sum += metrics.Recall
            loss_sum += loss_all.item()
    elif mode == 'test':
        model.eval()
        batch_size = len(graph.test_mask)
        steps = math.ceil(len(graph.test_mask) / batch_size)

        with torch.no_grad():
            for step in range(steps):
                if step == steps - 1:
                    edge_id = graph.test_mask[step * batch_size:]
                else:
                    edge_id = graph.test_mask[step * batch_size: step * batch_size + batch_size]

                output = model(graph, edge_id)

                label = graph.edge_label[edge_id].to(device)
                loss_all = loss_func(output, label)

                m = nn.Sigmoid()
                output = (m(output) > 0.5).type(torch.FloatTensor)
                metrics = Metrictor_PPI(output.cpu().data, label.cpu().data)
                metrics.show_result()

                f1_sum += metrics.F1
                accuracy_sum += metrics.Accuracy
                precision_sum += metrics.Precision
                recall_sum += metrics.Recall
                loss_sum += loss_all.item()

    f1 = f1_sum / steps
    accuracy = accuracy_sum / steps
    precision = precision_sum / steps
    recall = recall_sum / steps
    loss = loss_sum / steps

    if mode == 'train':
        scheduler.step(loss)

    return loss, f1
