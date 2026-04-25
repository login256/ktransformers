import argparse
import os

import torch

from kt_kernel import kt_kernel_ext
from kt_kernel_ext.moe import MOEConfig, PagedMoe_MOE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-root", default=os.getenv("PAGEDMOE_STORAGE_ROOT"), required=False)
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--qlen", type=int, default=1)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=1024)
    parser.add_argument("--cache-size-gib", type=float, default=0.25)
    parser.add_argument("--codebook-workers", type=int, default=1)
    parser.add_argument("--bitplane-workers", type=int, default=1)
    parser.add_argument("--compute-threads", type=int, default=1)
    args = parser.parse_args()

    if not args.storage_root:
        raise SystemExit("--storage-root or PAGEDMOE_STORAGE_ROOT is required")

    worker_config = kt_kernel_ext.WorkerPoolConfig()
    worker_config.subpool_count = 1
    worker_config.subpool_numa_map = [0]
    worker_config.subpool_thread_count = [0]
    worker_config.create_worker_threads = False
    cpu_infer = kt_kernel_ext.CPUInfer(worker_config)

    gpu_mask = torch.zeros(args.num_experts, dtype=torch.bool, device="cpu")
    config = MOEConfig(
        args.num_experts,
        args.topk,
        args.hidden_size,
        args.intermediate_size,
        gpu_mask.data_ptr(),
    )
    config.layer_idx = args.layer_id
    config.pool = cpu_infer.backend_
    config.path = args.storage_root
    config.max_len = args.qlen
    config.pagedmoe_cache_size_bytes = int(args.cache_size_gib * 1024 * 1024 * 1024)
    config.pagedmoe_codebook_workers = args.codebook_workers
    config.pagedmoe_bitplane_workers = args.bitplane_workers
    config.pagedmoe_compute_threads = args.compute_threads
    config.pagedmoe_pin_compute_workers = 0

    moe = PagedMoe_MOE(config)
    cpu_infer.submit(moe.load_weights_task())
    cpu_infer.sync()

    hidden = torch.randn((args.qlen, args.hidden_size), dtype=torch.bfloat16, device="cpu")
    expert_ids = torch.arange(args.qlen * args.topk, dtype=torch.long, device="cpu").reshape(args.qlen, args.topk)
    expert_ids.remainder_(args.num_experts)
    weights = torch.full((args.qlen, args.topk), 1.0 / args.topk, dtype=torch.float32, device="cpu")
    output = torch.empty_like(hidden)
    qlen_tensor = torch.tensor([args.qlen], dtype=torch.int32, device="cpu")

    cpu_infer.submit(
        moe.forward_task(
            qlen_tensor.data_ptr(),
            args.topk,
            expert_ids.data_ptr(),
            weights.data_ptr(),
            hidden.data_ptr(),
            output.data_ptr(),
        )
    )
    cpu_infer.sync()

    output_f32 = output.float()
    print(
        "pagedmoe forward ok:",
        {
            "shape": tuple(output.shape),
            "dtype": str(output.dtype),
            "finite": bool(torch.isfinite(output_f32).all().item()),
            "mean": float(output_f32.mean().item()),
            "std": float(output_f32.std().item()),
        },
    )


if __name__ == "__main__":
    main()
