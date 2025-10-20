import unittest

import torch
import torch.distributed as dist
from flash_attn import flash_attn_func

from llava.train.sequence_parallel.globals import get_ulysses_sp_pg, set_pg_manager
from llava.train.sequence_parallel.ulysses_attn import UlyssesAttention


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


class TestUlyssesAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dist.init_process_group("nccl")
        cls.rank = dist.get_rank()
        cls.world_size = dist.get_world_size()
        cls.dtype = torch.bfloat16
        cls.device = torch.device(f"cuda:{cls.rank}")

        cls.batch_size = 2
        cls.seqlen = 3816
        cls.nheads = 8
        cls.d = 128
        cls.dropout_p = 0
        cls.causal = True
        cls.deterministic = False

        assert cls.seqlen % cls.world_size == 0
        assert cls.d % 8 == 0

        cls.q = torch.randn(
            cls.batch_size, cls.seqlen, cls.nheads, cls.d, device=cls.device, dtype=cls.dtype, requires_grad=True
        )
        cls.k = torch.randn(
            cls.batch_size, cls.seqlen, cls.nheads, cls.d, device=cls.device, dtype=cls.dtype, requires_grad=True
        )
        cls.v = torch.randn(
            cls.batch_size, cls.seqlen, cls.nheads, cls.d, device=cls.device, dtype=cls.dtype, requires_grad=True
        )
        cls.dout = torch.randn(cls.batch_size, cls.seqlen, cls.nheads, cls.d, device=cls.device, dtype=cls.dtype)

        dist.broadcast(cls.q, src=0)
        dist.broadcast(cls.k, src=0)
        dist.broadcast(cls.v, src=0)
        dist.broadcast(cls.dout, src=0)

        cls.local_q = cls.q.chunk(cls.world_size, dim=1)[cls.rank].detach().clone()
        cls.local_q.requires_grad = True
        cls.local_k = cls.k.chunk(cls.world_size, dim=1)[cls.rank].detach().clone()
        cls.local_k.requires_grad = True
        cls.local_v = cls.v.chunk(cls.world_size, dim=1)[cls.rank].detach().clone()
        cls.local_v.requires_grad = True

        cls.local_dout = cls.dout.chunk(cls.world_size, dim=1)[cls.rank].detach().clone()

        set_pg_manager(dist.get_world_size(), -1)

        cls.dist_attn = UlyssesAttention(flash_attn_func, get_ulysses_sp_pg())

    def test_attention(self):
        if self.rank == 0:
            print("#" * 30)
            print("# ds-ulysses forward:")
            print("#" * 30)

        local_out = self.dist_attn(
            self.local_q,
            self.local_k,
            self.local_v,
            dropout_p=self.dropout_p,
            causal=self.causal,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=self.deterministic,
            return_attn_probs=True,
        )

        if self.rank == 0:
            print("#" * 30)
            print("# ds-ulysses backward:")
            print("#" * 30)

        local_out.backward(self.local_dout)

        dist.barrier()

        if self.rank == 0:
            print("#" * 30)
            print("# local forward:")
            print("#" * 30)

        out_ref, _, _ = flash_attn_func(
            self.q,
            self.k,
            self.v,
            dropout_p=self.dropout_p,
            causal=self.causal,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=self.deterministic,
            return_attn_probs=True,
        )

        if self.rank == 0:
            print("#" * 30)
            print("# local backward:")
            print("#" * 30)

        out_ref.backward(self.dout)

        dist.barrier()

        local_out_ref = out_ref.chunk(self.world_size, dim=1)[self.rank]

        log("out", local_out, rank0_only=True)
        log("out diff", local_out_ref - local_out)

        local_dq_ref = self.q.grad.chunk(self.world_size, dim=1)[self.rank]
        log("dq diff", local_dq_ref - self.local_q.grad)

        local_dk_ref = self.k.grad.chunk(self.world_size, dim=1)[self.rank]
        log("dk diff", local_dk_ref - self.local_k.grad)

        local_dv_ref = self.v.grad.chunk(self.world_size, dim=1)[self.rank]
        log("dv diff", local_dv_ref - self.local_v.grad)

        self.assertTrue(torch.allclose(local_out_ref, local_out, atol=1e-2))
        self.assertTrue(torch.allclose(local_dq_ref, self.local_q.grad, atol=1e-2))
        self.assertTrue(torch.allclose(local_dk_ref, self.local_k.grad, atol=1e-2))
        self.assertTrue(torch.allclose(local_dv_ref, self.local_v.grad, atol=1e-2))


if __name__ == "__main__":
    unittest.main()
