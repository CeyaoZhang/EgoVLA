import random
import unittest

import torch
import torch.distributed as dist
from flash_attn import flash_attn_varlen_qkvpacked_func

from llava.train.sequence_parallel.ring import ring_flash_attn_varlen_func, ring_flash_attn_varlen_qkvpacked_func


def set_seed(rank, seed=42):
    seed = rank + seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(msg, a, rank0_only=False):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if rank0_only:
        if rank == 0:
            print(
                f"{msg}: " f"max {a.abs().max().item()}, " f"mean {a.abs().mean().item()}",
                flush=True,
            )
        return

    for i in range(world_size):
        if i == rank:
            if rank == 0:
                print(f"{msg}:")
            print(
                f"[{rank}] " f"max {a.abs().max().item()}, " f"mean {a.abs().mean().item()}",
                flush=True,
            )
        dist.barrier()


def extract_local(value, cu_seqlens, rank, world_size):
    local_values = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        local_value = value[start:end].chunk(world_size, dim=0)[rank].detach().clone()
        local_values.append(local_value)
    return torch.cat(local_values, dim=0).contiguous()


def extract_lse(lse, cu_seqlens):
    values = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        value = lse[i, :, : end - start]
        values.append(value)
    return values


class TestRingAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dist.init_process_group("nccl")
        cls.rank = dist.get_rank()
        cls.world_size = dist.get_world_size()
        cls.dtype = torch.bfloat16
        cls.device = torch.device(f"cuda:{cls.rank}")
        set_seed(cls.rank)

        cls.batch_size = 1
        cls.nheads = 32
        cls.d = 128
        cls.dropout_p = 0
        cls.causal = True
        cls.deterministic = False

    def run_attention_test(self, cu_seqlens):
        cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32, device=self.device)
        max_seqlen = (cu_seqlens_tensor[1:] - cu_seqlens_tensor[:-1]).max().item()
        total_length = cu_seqlens[-1]
        num_seq = len(cu_seqlens) - 1

        print(f"seqlens: {cu_seqlens_tensor[1:] - cu_seqlens_tensor[:-1]}, max_seqlen: {max_seqlen}")

        assert self.d % 8 == 0

        qkv = torch.randn(
            total_length, 3, self.nheads, self.d, device=self.device, dtype=self.dtype, requires_grad=True
        )
        dist.broadcast(qkv, src=0)

        dout = torch.randn(total_length, self.nheads, self.d, device=self.device, dtype=self.dtype)
        dist.broadcast(dout, src=0)

        local_cu_seqlens_tensor = cu_seqlens_tensor // self.world_size
        local_max_seqlen = max_seqlen // self.world_size

        local_qkv = extract_local(qkv, cu_seqlens, self.rank, self.world_size)
        local_qkv.requires_grad = True
        local_dout = extract_local(dout, cu_seqlens, self.rank, self.world_size)

        dist.barrier()
        if self.rank == 0:
            print("#" * 30)
            print("# forward:")
            print("#" * 30)

        out, lse, _ = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens_tensor,
            max_seqlen,
            dropout_p=self.dropout_p,
            causal=self.causal,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=self.deterministic,
            return_attn_probs=True,
        )

        local_out = extract_local(out, cu_seqlens, self.rank, self.world_size)
        lse_list = extract_lse(lse, cu_seqlens)

        ring_out, ring_lse, _ = ring_flash_attn_varlen_qkvpacked_func(
            local_qkv,
            local_cu_seqlens_tensor,
            local_max_seqlen,
            dropout_p=self.dropout_p,
            causal=self.causal,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=self.deterministic,
            return_attn_probs=True,
        )

        ring_lse_list = extract_lse(ring_lse, local_cu_seqlens_tensor.tolist())

        log("out", out, rank0_only=True)
        log("out diff", local_out - ring_out)

        for lse, ring_lse in zip(lse_list, ring_lse_list):
            local_lse = lse.chunk(self.world_size, dim=-1)[self.rank]
            log("lse", lse, rank0_only=True)
            log("lse diff", local_lse - ring_lse)

        dist.barrier()
        if self.rank == 0:
            print("#" * 30)
            print("# backward:")
            print("#" * 30)

        out.backward(dout)
        dqkv = qkv.grad
        local_dqkv = extract_local(dqkv, cu_seqlens, self.rank, self.world_size)

        ring_out.backward(local_dout)
        ring_dqkv = local_qkv.grad

        log("dq diff", local_dqkv[:, 0] - ring_dqkv[:, 0])
        log("dk diff", local_dqkv[:, 1] - ring_dqkv[:, 1])
        log("dv diff", local_dqkv[:, 2] - ring_dqkv[:, 2])

        self.assertTrue(torch.allclose(local_out, ring_out, atol=1e-1))
        self.assertTrue(torch.allclose(local_dqkv[:, 0], ring_dqkv[:, 0], atol=1e-1))
        self.assertTrue(torch.allclose(local_dqkv[:, 1], ring_dqkv[:, 1], atol=1e-1))
        self.assertTrue(torch.allclose(local_dqkv[:, 2], ring_dqkv[:, 2], atol=1e-1))

    def test_attention_cases(self):
        cu_seqlens_list = [
            [0, 1024, 2176, 4352],
            # [0, 1768, 3536, 5200, 5660, 5856, 6052, 6252]
        ]
        for cu_seqlens in cu_seqlens_list:
            with self.subTest(cu_seqlens=cu_seqlens):
                self.run_attention_test(cu_seqlens)


if __name__ == "__main__":
    unittest.main()
