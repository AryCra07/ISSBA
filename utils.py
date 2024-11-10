# utils.py
import argparse

import torch


def get_parser():
    parser = argparse.ArgumentParser(
        description='Implementation of "Badnets: Identifying vulnerabilities in the machine learning model supply '
                    'chain".')
    parser.add_argument('--dataset', default='CIFAR10',
                        help='Which dataset to use (CIFAR10, default: CIFAR10)')
    parser.add_argument('--output_classes', default=10, type=int, help='number of the classification types')
    parser.add_argument('--load_local', action='store_true', help='Load a locally trained model (default: false)')
    parser.add_argument('--optimizer', default='sgd', help='Select the optimizer to use (sgd or adamw, default: sgd)')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to train the model, default: 10')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for the dataset, default: 64')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for the dataset, default: 0')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for the model, default: 0.01')
    parser.add_argument('--download', action='store_true', help='Download the dataset (default: false)')
    parser.add_argument('--data_path', default='./data/', help='Path to the dataset (default: ./data/)')
    parser.add_argument('--device', default='cpu',
                        help='Device to use for training/testing (cpu or cuda:1, default: cpu)')
    # Attack settings
    parser.add_argument('--poisoning_rate', type=float, default=0.1,
                        help='Poisoning rate (float, range from 0 to 1, default: 0.1)')
    parser.add_argument('--target_label', type=int, default=1,
                        help='Trigger label (int, range from 0 to 10, default: 1)')
    parser.add_argument('--trigger_path', default="./triggers/trigger_1.png",
                        help='Trigger path (default: ./triggers/trigger_2.png)')
    parser.add_argument('--trigger_size', type=int, default=5, help='Trigger size (int, default: 5)')

    return parser


def select_optimizer(optimization, param, lr):
    if optimization == 'adamw':
        optimizer = torch.optim.AdamW(param, lr=lr)
    elif optimization == 'sgd':
        optimizer = torch.optim.SGD(param, lr=lr)
    else:
        print('select adamw as optimizer automatically')
        optimizer = torch.optim.AdamW(param, lr=lr)
    return optimizer
