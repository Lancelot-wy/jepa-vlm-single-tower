import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from jepa_vlm.modeling.state_loss import DistributedRunningCenter


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _worker(rank, world, port, output):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    center = DistributedRunningCenter(2, 0.99)
    value = 1.0 if rank == 0 else 3.0
    target = torch.full((1, 1, 2), value)
    center.update(target, torch.ones(1, 1, dtype=torch.bool))
    output[rank] = center.running_center.tolist()
    dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_running_center_is_identical_across_two_ranks():
    manager = mp.Manager()
    output = manager.dict()
    mp.spawn(_worker, args=(2, _free_port(), output), nprocs=2, join=True)
    assert output[0] == output[1] == [2.0, 2.0]
