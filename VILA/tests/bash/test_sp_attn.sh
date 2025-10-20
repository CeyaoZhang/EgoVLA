SP_DEGREE=8

torchrun --nproc_per_node=$SP_DEGREE tests/seq_parallel/test_sp_ring_attn_varlen.py
torchrun --nproc_per_node=$SP_DEGREE tests/seq_parallel/test_sp_ulysses_attn.py
