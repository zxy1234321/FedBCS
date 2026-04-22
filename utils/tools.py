import os
import numpy as np
import torch


def create_if_not_exists(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def generate_online_clients_sequence(epochs, parti_num, online_ratio):
    sequence = {}
    for epoch in range(epochs):
        total_clients = list(range(parti_num))
        online_clients = np.random.choice(total_clients, int(parti_num * online_ratio), replace=False).tolist()
        sequence[epoch] = online_clients
    return sequence
