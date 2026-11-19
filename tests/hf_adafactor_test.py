import importlib.util
import os
import socket
import unittest
from datetime import timedelta
from unittest import TestCase

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Replicate, Shard
from transformers.optimization import Adafactor

# examples/hf_transformers is not a package; load its optimizer module from the repository root, the working
# directory these tests run from
OPTIM_MODULE_PATH = 'examples/hf_transformers/custom/optim.py'
# The settings of the XSum T5 configs: a fixed step size (distillation runs), and T5X's fine-tuning settings, where
# each update is scaled by its weight's RMS
ADAFACTOR_SETTINGS = {
    'fixed_step': dict(lr=3.0e-4, scale_parameter=False, relative_step=False, warmup_init=False),
    't5x': dict(lr=1.0e-3, scale_parameter=True, relative_step=False, warmup_init=False),
}
NUM_STEPS = 3
TOLERANCE = dict(rtol=0.0, atol=1.0e-6)


def load_optim_module():
    spec = importlib.util.spec_from_file_location('hf_transformers_custom_optim', OPTIM_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model():
    torch.manual_seed(0)
    # Odd row counts shard unevenly over two ranks; LayerNorm adds a 1D (unfactored) parameter
    return nn.Sequential(nn.Linear(16, 33, bias=False), nn.LayerNorm(33, bias=False), nn.Linear(33, 7, bias=False))


def train(model, optimizer_cls, device, optimizer_kwargs):
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
    for step in range(NUM_STEPS):
        generator = torch.Generator().manual_seed(100 + step)
        x = torch.randn(4, 16, generator=generator).to(device)
        model(x).pow(2).mean().backward()
        optimizer.step()
        optimizer.zero_grad()
    return optimizer


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def fsdp2_worker(rank, world_size, port, use_cuda):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    if use_cuda:
        device = torch.device('cuda', rank)
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend='nccl', rank=rank, world_size=world_size, timeout=timedelta(seconds=120), device_id=device
        )
    else:
        device = torch.device('cpu')
        dist.init_process_group(backend='gloo', rank=rank, world_size=world_size, timeout=timedelta(seconds=120))

    try:
        optim_module = load_optim_module()
        # Without an explicit mesh, fully_shard builds one on CUDA whenever CUDA is available, which would put the
        # gloo (CPU) case's shards on GPUs
        mesh = init_device_mesh(device.type, (world_size,))
        for setting_name, optimizer_kwargs in ADAFACTOR_SETTINGS.items():
            reference_model = build_model().to(device)
            train(reference_model, Adafactor, device, optimizer_kwargs)

            model = build_model().to(device)
            for layer in model:
                fully_shard(layer, mesh=mesh)
            fully_shard(model, mesh=mesh)
            optimizer = train(model, optim_module.DTensorAwareAdafactor, device, optimizer_kwargs)

            for (name, param), reference_param in zip(model.named_parameters(), reference_model.parameters()):
                assert isinstance(param, DTensor), f'rank {rank}, {setting_name}: {name} is not a DTensor'
                state = optimizer.state[param]
                if param.dim() >= 2:
                    assert tuple(state['exp_avg_sq_row'].placements) == (Shard(0),), f'{setting_name}: {name}'
                    assert tuple(state['exp_avg_sq_col'].placements) == (Replicate(),), f'{setting_name}: {name}'
                    assert 'exp_avg_sq' not in state
                else:
                    assert 'exp_avg_sq' in state
                assert torch.allclose(param.full_tensor(), reference_param, **TOLERANCE), \
                    f'rank {rank}, {setting_name}: {name} differs from single-process Adafactor'
        dist.barrier()
    finally:
        dist.destroy_process_group()


class DTensorAwareAdafactorTest(TestCase):
    def test_plain_tensors_match_hf_adafactor_exactly(self):
        optim_module = load_optim_module()
        device = torch.device('cpu')
        for setting_name, optimizer_kwargs in ADAFACTOR_SETTINGS.items():
            reference_model = build_model()
            train(reference_model, Adafactor, device, optimizer_kwargs)
            model = build_model()
            train(model, optim_module.DTensorAwareAdafactor, device, optimizer_kwargs)
            for param, reference_param in zip(model.parameters(), reference_model.parameters()):
                assert torch.equal(param, reference_param), setting_name

    def test_fsdp2_matches_single_process_on_cpu(self):
        mp.spawn(fsdp2_worker, args=(2, find_free_port(), False), nprocs=2, join=True)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, 'needs two CUDA devices')
    def test_fsdp2_matches_single_process_on_two_gpus(self):
        mp.spawn(fsdp2_worker, args=(2, find_free_port(), True), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
