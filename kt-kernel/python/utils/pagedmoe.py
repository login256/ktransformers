import os
from typing import List, Optional

import torch

from ..experts_base import BaseMoEWrapper
from kt_kernel import kt_kernel_ext
from kt_kernel_ext.moe import MOEConfig

try:
    from kt_kernel_ext.moe import PagedMoe_MOE

    _HAS_PAGEDMOE_SUPPORT = True
except (ImportError, AttributeError):
    PagedMoe_MOE = None
    _HAS_PAGEDMOE_SUPPORT = False


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _env_bool_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return 1 if value.strip().lower() in {"1", "on", "true", "yes", "y"} else 0


def _cache_size_bytes_from_env() -> int:
    if os.getenv("PAGEDMOE_CACHE_SIZE_BYTES"):
        return int(os.environ["PAGEDMOE_CACHE_SIZE_BYTES"])
    if os.getenv("PAGEDMOE_CACHE_SIZE_GIB"):
        return int(float(os.environ["PAGEDMOE_CACHE_SIZE_GIB"]) * 1024 * 1024 * 1024)
    return 0


class PagedMoeWrapper(BaseMoEWrapper):
    """
    pagedmoe-backed MoE wrapper.

    The KT side only schedules calls and moves tensors. The actual expert storage,
    read workers, cache, and compute worker pool live inside pagedmoe-capi.
    """

    _pagedmoe_cpu_infer_instance = None

    @classmethod
    def _get_cpu_infer(
        cls,
        cpuinfer_threads: int,
        threadpool_count: int,
        numa_nodes=None,
    ):
        if cls._pagedmoe_cpu_infer_instance is None:
            worker_config = kt_kernel_ext.WorkerPoolConfig()
            subpool_count = max(1, int(threadpool_count))
            if numa_nodes is not None:
                if len(numa_nodes) != subpool_count:
                    raise ValueError(
                        f"numa_nodes length ({len(numa_nodes)}) must match threadpool_count ({subpool_count})"
                    )
                subpool_numa_map = list(numa_nodes)
            else:
                subpool_numa_map = list(range(subpool_count))

            worker_config.subpool_count = subpool_count
            worker_config.subpool_numa_map = subpool_numa_map
            worker_config.subpool_thread_count = [0 for _ in range(subpool_count)]
            worker_config.create_worker_threads = False
            cls._pagedmoe_cpu_infer_instance = kt_kernel_ext.CPUInfer(worker_config)

        return cls._pagedmoe_cpu_infer_instance

    def __init__(
        self,
        layer_idx: int,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        moe_intermediate_size: int,
        gpu_experts_mask: Optional[torch.Tensor],
        cpuinfer_threads: int,
        threadpool_count: int,
        weight_path: str,
        chunked_prefill_size: int,
        cpu_save: bool = False,
        max_deferred_experts_per_token: Optional[int] = None,
        method: str = "PAGEDMOE",
        numa_nodes: Optional[List[int]] = None,
    ):
        if not _HAS_PAGEDMOE_SUPPORT:
            raise RuntimeError(
                "PagedMoE backend not available. Rebuild kt_kernel_ext with "
                "CPUINFER_ENABLE_PAGEDMOE=ON and CPUINFER_PAGEDMOE_ROOT=/path/to/pagedmoe."
            )
        if cpu_save:
            raise ValueError("PAGEDMOE does not support KTransformers cpu_save weight conversion")
        if max_deferred_experts_per_token not in (None, 0):
            raise ValueError("PAGEDMOE does not support KTransformers deferred expert execution")

        self.pagedmoe_cache_size_bytes = _cache_size_bytes_from_env()
        self.pagedmoe_codebook_workers = _env_int("PAGEDMOE_CODEBOOK_WORKERS", 1)
        self.pagedmoe_bitplane_workers = _env_int("PAGEDMOE_BITPLANE_WORKERS", 4)
        self.pagedmoe_compute_threads = _env_int("PAGEDMOE_COMPUTE_THREADS", max(1, int(cpuinfer_threads)))
        self.pagedmoe_pin_compute_workers = _env_bool_int("PAGEDMOE_PIN_COMPUTE_WORKERS", 0)
        self.pagedmoe_num_layers = _env_int("PAGEDMOE_NUM_LAYERS", 0)
        self.pagedmoe_num_blocks = _env_int("PAGEDMOE_NUM_BLOCKS", 0)

        super().__init__(
            layer_idx=layer_idx,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            gpu_experts_mask=gpu_experts_mask,
            cpuinfer_threads=cpuinfer_threads,
            threadpool_count=threadpool_count,
            weight_path=weight_path,
            chunked_prefill_size=chunked_prefill_size,
            cpu_save=cpu_save,
            max_deferred_experts_per_token=0,
            method=method,
            numa_nodes=numa_nodes,
        )

    def load_weights_from_tensors(
        self,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        physical_to_logical_map_cpu: torch.Tensor,
    ):
        raise NotImplementedError("PAGEDMOE uses pre-exported pagedmoe storage; call load_weights() instead")

    def load_weights(self, physical_to_logical_map_cpu: Optional[torch.Tensor] = None):
        moe_config = MOEConfig(
            self.num_experts,
            self.num_experts_per_tok,
            self.hidden_size,
            self.moe_intermediate_size,
            self.gpu_experts_mask.data_ptr(),
        )
        moe_config.layer_idx = self.layer_idx
        moe_config.pool = self.cpu_infer.backend_
        moe_config.max_len = self.chunked_prefill_size
        moe_config.path = self.weight_path

        moe_config.pagedmoe_cache_size_bytes = self.pagedmoe_cache_size_bytes
        moe_config.pagedmoe_codebook_workers = self.pagedmoe_codebook_workers
        moe_config.pagedmoe_bitplane_workers = self.pagedmoe_bitplane_workers
        moe_config.pagedmoe_compute_threads = self.pagedmoe_compute_threads
        moe_config.pagedmoe_pin_compute_workers = self.pagedmoe_pin_compute_workers
        moe_config.pagedmoe_num_layers = self.pagedmoe_num_layers
        moe_config.pagedmoe_num_blocks = self.pagedmoe_num_blocks

        self.moe = PagedMoe_MOE(moe_config)
        self.cpu_infer.submit(self.moe.load_weights_task())
        self.cpu_infer.sync()
