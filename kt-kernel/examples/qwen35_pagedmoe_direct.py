#!/usr/bin/env python3
from __future__ import annotations

"""
Run Qwen3.5 MoE with Transformers for the model/decode loop, pagedmoe-python
for routed expert execution, and kt-kernel for non-routed shared expert MLP /
shared expert gate execution.

Transformers Qwen3.5 model
  -> patched layer.mlp.experts
    -> pagedmoe-python ExpertRuntime
  -> optional patched layer.mlp.shared_expert/shared_expert_gate
    -> kt_kernel_ext.mlp.MLP / kt_kernel_ext.linear.Linear

The old kt-kernel PAGEDMOE wrapper path is still available with
--routed-backend kt-wrapper.
"""

import argparse
import json
import os
import pathlib
import sys
import time
from dataclasses import dataclass, field
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
import transformers.utils.import_utils as hf_import_utils

# Keep this path deterministic on machines with a mismatched optional
# causal-conv1d install. Qwen3.5 has a torch fallback.
hf_import_utils.is_causal_conv1d_available = lambda: False

from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: E402
    Qwen3_5MoeForConditionalGeneration,
)


def add_kt_kernel_source_path() -> None:
    kt_kernel_root = pathlib.Path(__file__).resolve().parents[1]
    kt_python = kt_kernel_root / "python"
    if str(kt_python) not in sys.path:
        sys.path.insert(0, str(kt_python))


def add_pagedmoe_python_path() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[4]
    for build_kind in ("release", "debug"):
        ext_dir = repo_root / "src" / "target" / build_kind
        if ext_dir.exists() and str(ext_dir) not in sys.path:
            sys.path.insert(0, str(ext_dir))
            return


add_kt_kernel_source_path()
add_pagedmoe_python_path()
from kt_kernel import KTMoEWrapper, kt_kernel_ext  # noqa: E402
import libktensor_python as pagedmoe_py  # noqa: E402


QWEN35_HIDDEN_SIZE = 4096
QWEN35_TOPK = 10
QWEN35_NUM_LAYERS = 60
GGML_FP32 = kt_kernel_ext.kvcache.ggml_type.FP32
PAGEDMOE_TARGET_PRECISIONS = tuple(
    pagedmoe_py.expert.ExpertRuntime.default_target_precisions()
    if hasattr(pagedmoe_py.expert.ExpertRuntime, "default_target_precisions")
    else ()
)


def load_weight_map(model_root: pathlib.Path) -> dict[str, str]:
    index_path = model_root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"weight_map missing from {index_path}")
    return {str(key): str(value) for key, value in weight_map.items()}


def is_routed_expert_param(name: str) -> bool:
    return ".mlp.experts.gate_up_proj" in name or ".mlp.experts.down_proj" in name


def resolve_device_for_param(
    device_map: dict[str, int | str | torch.device],
    param_name: str,
) -> int | str | torch.device:
    module_name = param_name
    while len(module_name) > 0 and module_name not in device_map:
        module_name = ".".join(module_name.split(".")[:-1])
    if module_name in device_map:
        return device_map[module_name]
    if "" in device_map:
        return device_map[""]
    raise ValueError(f"{param_name} does not have any device assignment")


def load_non_routed_weights(
    model: nn.Module,
    model_root: pathlib.Path,
    device_map: dict[str, int | str | torch.device],
) -> None:
    index = load_weight_map(model_root)
    model_keys = set(model.state_dict().keys())
    grouped: dict[str, list[str]] = {}
    skipped_routed = 0
    for key, shard_name in index.items():
        if is_routed_expert_param(key):
            skipped_routed += 1
            continue
        if key not in model_keys:
            continue
        grouped.setdefault(shard_name, []).append(key)

    loaded = 0
    for shard_name, keys in sorted(grouped.items()):
        shard_path = model_root / shard_name
        with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
            for key in keys:
                tensor = shard.get_tensor(key)
                target_device = resolve_device_for_param(device_map, key)
                if target_device == "disk":
                    raise ValueError("disk offload is not supported in this runtime loader")
                set_module_tensor_to_device(
                    model,
                    key,
                    target_device,
                    value=tensor,
                    dtype=torch.bfloat16,
                )
                loaded += 1

    print(
        "[setup] "
        f"loaded_non_routed_parameters={loaded} "
        f"skipped_routed_parameters={skipped_routed}",
        flush=True,
    )


def build_device_map(
    model: nn.Module,
    requested: str | dict[str, Any],
) -> dict[str, int | str | torch.device]:
    if isinstance(requested, dict):
        return requested
    if requested == "auto":
        return infer_auto_device_map(
            model,
            no_split_module_classes=getattr(model, "_no_split_modules", None),
            dtype=torch.bfloat16,
        )
    return {"": requested}


@dataclass
class RouteStats:
    token_calls: int = 0
    route_slots: int = 0
    route_unique_requests: int = 0
    submitted_reads: int = 0
    completed_reads: int = 0
    submitted_compute_tasks: int = 0
    completed_compute_tasks: int = 0
    per_layer_unique_experts: dict[int, set[int]] = field(default_factory=dict)

    def reset(self) -> None:
        self.token_calls = 0
        self.route_slots = 0
        self.route_unique_requests = 0
        self.submitted_reads = 0
        self.completed_reads = 0
        self.submitted_compute_tasks = 0
        self.completed_compute_tasks = 0
        self.per_layer_unique_experts.clear()

    def record(self, layer_idx: int, topk_ids: torch.Tensor) -> None:
        ids = topk_ids.detach().to(device="cpu", dtype=torch.long)
        flat = ids.reshape(-1)
        unique = torch.unique(flat)
        self.token_calls += int(ids.shape[0])
        self.route_slots += int(ids.numel())
        self.route_unique_requests += int(unique.numel())
        layer_set = self.per_layer_unique_experts.setdefault(layer_idx, set())
        layer_set.update(int(x) for x in unique.tolist())

    def record_pagedmoe_stats(self, stats: Any) -> None:
        self.submitted_reads += int(stats.submitted_reads)
        self.completed_reads += int(stats.completed_reads)
        self.submitted_compute_tasks += int(stats.submitted_compute_tasks)
        self.completed_compute_tasks += int(stats.completed_compute_tasks)

    def render(self) -> str:
        unique_total = sum(len(items) for items in self.per_layer_unique_experts.values())
        return (
            "route_stats "
            f"token_calls={self.token_calls} "
            f"route_slots={self.route_slots} "
            f"route_unique_requests={self.route_unique_requests} "
            f"unique_experts_across_layers={unique_total} "
            f"submitted_reads={self.submitted_reads} "
            f"completed_reads={self.completed_reads} "
            f"submitted_compute_tasks={self.submitted_compute_tasks} "
            f"completed_compute_tasks={self.completed_compute_tasks}"
        )


class KTPagedMoeQwen35Experts(nn.Module):
    def __init__(
        self,
        *,
        layer_idx: int,
        num_experts: int,
        topk: int,
        hidden_size: int,
        intermediate_size: int,
        storage_root: pathlib.Path,
        compute_threads: int,
        threadpool_count: int,
        chunked_prefill_size: int,
        route_stats: RouteStats,
        numa_nodes: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.topk = topk
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.storage_root = storage_root
        self.compute_threads = compute_threads
        self.threadpool_count = threadpool_count
        self.chunked_prefill_size = chunked_prefill_size
        self.route_stats = route_stats
        self.numa_nodes = numa_nodes
        self.wrapper = None

    def initialize_backend(self) -> None:
        if self.wrapper is not None:
            return
        gpu_experts_mask = torch.zeros(self.num_experts, dtype=torch.bool, device="cpu")
        self.wrapper = KTMoEWrapper(
            layer_idx=self.layer_idx,
            num_experts=self.num_experts,
            num_experts_per_tok=self.topk,
            hidden_size=self.hidden_size,
            moe_intermediate_size=self.intermediate_size,
            gpu_experts_mask=gpu_experts_mask,
            cpuinfer_threads=self.compute_threads,
            threadpool_count=self.threadpool_count,
            weight_path=str(self.storage_root),
            chunked_prefill_size=self.chunked_prefill_size,
            method="PAGEDMOE",
            numa_nodes=self.numa_nodes,
        )
        self.wrapper.load_weights()

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.wrapper is None:
            raise RuntimeError("KTPagedMoeQwen35Experts backend was not initialized")
        if hidden_states.device.type != "cuda":
            raise RuntimeError("PAGEDMOE kt-kernel path expects hidden_states on CUDA")
        if hidden_states.dtype != torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)

        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
        topk_ids = top_k_index.reshape(-1, top_k_index.shape[-1]).contiguous()
        topk_weights = top_k_weights.reshape(-1, top_k_weights.shape[-1]).contiguous()
        self.route_stats.record(self.layer_idx, topk_ids)

        output = self.wrapper.forward(
            flat_hidden,
            topk_ids,
            topk_weights,
            torch.cuda.current_stream(flat_hidden.device).cuda_stream,
        )
        return output.reshape_as(hidden_states)


class PagedMoePythonQwen35Experts(nn.Module):
    def __init__(
        self,
        *,
        layer_idx: int,
        topk: int,
        hidden_size: int,
        runtime: Any,
        route_stats: RouteStats,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.topk = topk
        self.hidden_size = hidden_size
        self.runtime = runtime
        self.route_stats = route_stats
        self._expert_outputs: torch.Tensor | None = None
        self._output_ptrs: np.ndarray | None = None

    def _output_buffers(self, topk: int) -> tuple[torch.Tensor, np.ndarray]:
        if (
            self._expert_outputs is None
            or self._expert_outputs.shape[0] < topk
            or self._expert_outputs.shape[1] != self.hidden_size
        ):
            self._expert_outputs = torch.empty((topk, self.hidden_size), dtype=torch.float32, device="cpu")
            self._output_ptrs = np.asarray(
                [self._expert_outputs[i].data_ptr() for i in range(topk)],
                dtype=np.uint64,
            )
        if self._output_ptrs is None:
            raise RuntimeError("pagedmoe-python output pointer buffer was not initialized")
        return self._expert_outputs, self._output_ptrs[:topk]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.device.type != "cuda":
            raise RuntimeError("pagedmoe-python routed expert path expects hidden_states on CUDA")
        if hidden_states.dtype != torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)

        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
        topk_ids = top_k_index.reshape(-1, top_k_index.shape[-1]).contiguous()
        topk_weights = top_k_weights.reshape(-1, top_k_weights.shape[-1]).contiguous()
        self.route_stats.record(self.layer_idx, topk_ids)

        hidden_cpu = flat_hidden.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        topk_ids_cpu = topk_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
        topk_weights_cpu = topk_weights.detach().to(device="cpu", dtype=torch.float32).contiguous()
        topk_ids_np = topk_ids_cpu.numpy()
        topk_weights_np = topk_weights_cpu.numpy()

        num_tokens = int(hidden_cpu.shape[0])
        final_hidden_cpu = torch.empty((num_tokens, self.hidden_size), dtype=torch.float32, device="cpu")
        topk = int(topk_ids_np.shape[1])
        expert_outputs, output_ptrs = self._output_buffers(topk)

        for token_idx in range(num_tokens):
            expert_ids_np = np.asarray(topk_ids_np[token_idx], dtype=np.uint16)
            router_weights_np = np.asarray(topk_weights_np[token_idx], dtype=np.float32)
            stats = self.runtime.execute(
                layer_index=self.layer_idx,
                activations_ptr=hidden_cpu[token_idx].data_ptr(),
                expert_ids=expert_ids_np,
                router_weights=router_weights_np,
                output_ptrs=output_ptrs,
            )
            self.route_stats.record_pagedmoe_stats(stats)
            weighted = expert_outputs[:topk] * topk_weights_cpu[token_idx].unsqueeze(1)
            final_hidden_cpu[token_idx].copy_(weighted.sum(dim=0))

        return final_hidden_cpu.to(device=flat_hidden.device, dtype=hidden_states.dtype).reshape_as(hidden_states)


class KTDenseCPUInfer:
    _instance: Any = None
    _threads: int | None = None

    @classmethod
    def get(cls, threads: int) -> Any:
        threads = max(1, int(threads))
        if cls._instance is None:
            cls._instance = kt_kernel_ext.CPUInfer(threads)
            cls._threads = threads
        elif cls._threads != threads:
            print(
                "[setup] "
                f"reusing kt dense CPUInfer with threads={cls._threads}; "
                f"requested_threads={threads}",
                flush=True,
            )
        return cls._instance


def _cpu_fp32_weight(weight: torch.Tensor) -> torch.Tensor:
    return weight.detach().to(device="cpu", dtype=torch.float32).contiguous()


class KTCPUMLP(nn.Module):
    def __init__(
        self,
        original_mlp: nn.Module,
        *,
        dense_threads: int,
        stride: int,
        group_max_len: int,
        debug_name: str = "kt-mlp",
        debug_finite: bool = False,
    ) -> None:
        super().__init__()
        self.debug_name = debug_name
        self.debug_finite = debug_finite
        self.hidden_size = int(original_mlp.gate_proj.in_features)
        self.intermediate_size = int(original_mlp.gate_proj.out_features)
        self.cpu_infer = KTDenseCPUInfer.get(dense_threads)

        self.gate_proj_weight = _cpu_fp32_weight(original_mlp.gate_proj.weight)
        self.up_proj_weight = _cpu_fp32_weight(original_mlp.up_proj.weight)
        self.down_proj_weight = _cpu_fp32_weight(original_mlp.down_proj.weight)

        config = kt_kernel_ext.mlp.MLPConfig(
            self.hidden_size,
            self.intermediate_size,
            int(stride),
            int(group_max_len),
            self.gate_proj_weight.data_ptr(),
            self.up_proj_weight.data_ptr(),
            self.down_proj_weight.data_ptr(),
            GGML_FP32,
            GGML_FP32,
            GGML_FP32,
            GGML_FP32,
        )
        self.mlp = kt_kernel_ext.mlp.MLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, original_shape[-1])
        if flat_hidden.device.type != "cuda":
            raise RuntimeError("kt-kernel shared expert path expects hidden_states on CUDA")
        if self.debug_finite and not torch.isfinite(flat_hidden).all():
            raise RuntimeError(f"non-finite input to {self.debug_name}")

        input_cpu = flat_hidden.detach().to(device="cpu", dtype=torch.float32).contiguous()
        output_cpu = torch.empty(
            (input_cpu.shape[0], self.hidden_size),
            dtype=torch.float32,
            device="cpu",
        )
        self.cpu_infer.submit(
            self.mlp.forward_task(
                int(input_cpu.shape[0]),
                input_cpu.data_ptr(),
                output_cpu.data_ptr(),
            )
        )
        self.cpu_infer.sync()
        if self.debug_finite and not torch.isfinite(output_cpu).all():
            finite = output_cpu[torch.isfinite(output_cpu)]
            finite_min = float(finite.min().item()) if finite.numel() > 0 else float("nan")
            finite_max = float(finite.max().item()) if finite.numel() > 0 else float("nan")
            raise RuntimeError(
                f"non-finite output from {self.debug_name}: "
                f"finite_min={finite_min} finite_max={finite_max}"
            )
        return output_cpu.to(device=flat_hidden.device, dtype=hidden_states.dtype).reshape(original_shape)


class KTCPULinear(nn.Module):
    def __init__(
        self,
        original_linear: nn.Linear,
        *,
        dense_threads: int,
        stride: int,
        group_max_len: int,
        debug_name: str = "kt-linear",
        debug_finite: bool = False,
    ) -> None:
        super().__init__()
        self.debug_name = debug_name
        self.debug_finite = debug_finite
        self.input_size = int(original_linear.in_features)
        self.output_size = int(original_linear.out_features)
        self.cpu_infer = KTDenseCPUInfer.get(dense_threads)
        self.proj_weight = _cpu_fp32_weight(original_linear.weight)
        effective_stride = max(1, min(int(stride), self.output_size))

        config = kt_kernel_ext.linear.LinearConfig(
            self.input_size,
            self.output_size,
            effective_stride,
            int(group_max_len),
            self.proj_weight.data_ptr(),
            GGML_FP32,
            GGML_FP32,
        )
        self.linear = kt_kernel_ext.linear.Linear(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, original_shape[-1])
        if flat_hidden.device.type != "cuda":
            raise RuntimeError("kt-kernel linear path expects hidden_states on CUDA")
        if self.debug_finite and not torch.isfinite(flat_hidden).all():
            raise RuntimeError(f"non-finite input to {self.debug_name}")

        input_cpu = flat_hidden.detach().to(device="cpu", dtype=torch.float32).contiguous()
        output_cpu = torch.empty(
            (input_cpu.shape[0], self.output_size),
            dtype=torch.float32,
            device="cpu",
        )
        self.cpu_infer.submit(
            self.linear.forward_task(
                int(input_cpu.shape[0]),
                input_cpu.data_ptr(),
                output_cpu.data_ptr(),
            )
        )
        self.cpu_infer.sync()
        if self.debug_finite and not torch.isfinite(output_cpu).all():
            finite = output_cpu[torch.isfinite(output_cpu)]
            finite_min = float(finite.min().item()) if finite.numel() > 0 else float("nan")
            finite_max = float(finite.max().item()) if finite.numel() > 0 else float("nan")
            raise RuntimeError(
                f"non-finite output from {self.debug_name}: "
                f"finite_min={finite_min} finite_max={finite_max}"
            )
        return output_cpu.to(device=flat_hidden.device, dtype=hidden_states.dtype).reshape(
            *original_shape[:-1],
            self.output_size,
        )


@dataclass
class DirectRuntimeConfig:
    model_root: pathlib.Path
    storage_root: pathlib.Path
    routed_backend: str = "pagedmoe-python"
    cache_size_gib: float = 48.0
    codebook_workers: int = 1
    bitplane_workers: int = 4
    compute_threads: int = 8
    torch_threads: int = 1
    torch_interop_threads: int = 1
    threadpool_count: int = 1
    chunked_prefill_size: int = 4096
    kt_shared_expert: bool = False
    dense_threads: int = 8
    dense_stride: int = 32
    debug_finite: bool = False
    device_map: str | dict[str, Any] = "auto"
    max_new_tokens: int = 2

    @property
    def cache_size_bytes(self) -> int:
        return int(self.cache_size_gib * 1024**3)


class Qwen35KTPagedMoeModel:
    def __init__(self, cfg: DirectRuntimeConfig) -> None:
        self.cfg = cfg
        self.route_stats = RouteStats()
        self._configure_pagedmoe_env()
        self.pagedmoe_runtime = None
        if cfg.routed_backend == "pagedmoe-python":
            self.pagedmoe_runtime = pagedmoe_py.expert.ExpertRuntime(
                str(cfg.storage_root),
                cache_size_bytes=cfg.cache_size_bytes,
                codebook_workers=cfg.codebook_workers,
                bitplane_workers=cfg.bitplane_workers,
                compute_threads=cfg.compute_threads,
            )

        config = AutoConfig.from_pretrained(str(cfg.model_root), trust_remote_code=True)
        text_config = config.text_config
        self._validate_qwen35_config(text_config)

        with init_empty_weights():
            model = Qwen3_5MoeForConditionalGeneration(config)
            for layer_idx, layer in enumerate(model.model.language_model.layers):
                if cfg.routed_backend == "pagedmoe-python":
                    if self.pagedmoe_runtime is None:
                        raise RuntimeError("pagedmoe-python runtime was not initialized")
                    layer.mlp.experts = PagedMoePythonQwen35Experts(
                        layer_idx=layer_idx,
                        topk=int(text_config.num_experts_per_tok),
                        hidden_size=int(text_config.hidden_size),
                        runtime=self.pagedmoe_runtime,
                        route_stats=self.route_stats,
                    )
                elif cfg.routed_backend == "kt-wrapper":
                    layer.mlp.experts = KTPagedMoeQwen35Experts(
                        layer_idx=layer_idx,
                        num_experts=int(text_config.num_experts),
                        topk=int(text_config.num_experts_per_tok),
                        hidden_size=int(text_config.hidden_size),
                        intermediate_size=int(text_config.moe_intermediate_size),
                        storage_root=cfg.storage_root,
                        compute_threads=cfg.compute_threads,
                        threadpool_count=cfg.threadpool_count,
                        chunked_prefill_size=cfg.chunked_prefill_size,
                        route_stats=self.route_stats,
                    )
                else:
                    raise ValueError(f"unknown routed backend: {cfg.routed_backend}")
            model.tie_weights()

        device_map = build_device_map(model, cfg.device_map)
        load_non_routed_weights(model, cfg.model_root, device_map)
        self.model = dispatch_model(model, device_map=device_map)
        self.tokenizer = AutoTokenizer.from_pretrained(str(cfg.model_root), trust_remote_code=True)

        if cfg.kt_shared_expert:
            self._replace_shared_experts_with_kt()
        self._initialize_routed_backends()
        if cfg.debug_finite:
            self._install_finite_debug_hooks()
        self.model.eval()

    def _configure_pagedmoe_env(self) -> None:
        os.environ["PAGEDMOE_CACHE_SIZE_GIB"] = str(self.cfg.cache_size_gib)
        os.environ["PAGEDMOE_CODEBOOK_WORKERS"] = str(self.cfg.codebook_workers)
        os.environ["PAGEDMOE_BITPLANE_WORKERS"] = str(self.cfg.bitplane_workers)
        os.environ["PAGEDMOE_COMPUTE_THREADS"] = str(self.cfg.compute_threads)
        os.environ.setdefault("PAGEDMOE_PIN_COMPUTE_WORKERS", "0")
        os.environ["PAGEDMOE_NUM_LAYERS"] = str(QWEN35_NUM_LAYERS)

    @staticmethod
    def _validate_qwen35_config(text_config: Any) -> None:
        expected = {
            "hidden_size": QWEN35_HIDDEN_SIZE,
            "moe_intermediate_size": 1024,
            "num_hidden_layers": QWEN35_NUM_LAYERS,
            "num_experts_per_tok": QWEN35_TOPK,
        }
        for field_name, expected_value in expected.items():
            actual = int(getattr(text_config, field_name))
            if actual != expected_value:
                raise ValueError(f"{field_name}={actual} does not match expected {expected_value}")

    def _initialize_routed_backends(self) -> None:
        for layer_idx, layer in enumerate(self.model.model.language_model.layers):
            experts = layer.mlp.experts
            if self.cfg.routed_backend == "pagedmoe-python":
                if not isinstance(experts, PagedMoePythonQwen35Experts):
                    raise TypeError(f"layer {layer_idx} experts not patched: {type(experts).__name__}")
            elif self.cfg.routed_backend == "kt-wrapper":
                if not isinstance(experts, KTPagedMoeQwen35Experts):
                    raise TypeError(f"layer {layer_idx} experts not patched: {type(experts).__name__}")
                experts.initialize_backend()
            else:
                raise ValueError(f"unknown routed backend: {self.cfg.routed_backend}")
        print(f"[setup] initialized routed backend={self.cfg.routed_backend}", flush=True)

    def _replace_shared_experts_with_kt(self) -> None:
        for layer_idx, layer in enumerate(self.model.model.language_model.layers):
            layer.mlp.shared_expert = KTCPUMLP(
                layer.mlp.shared_expert,
                dense_threads=self.cfg.dense_threads,
                stride=self.cfg.dense_stride,
                group_max_len=self.cfg.chunked_prefill_size,
                debug_name=f"layer{layer_idx}.shared_expert",
                debug_finite=self.cfg.debug_finite,
            )
            layer.mlp.shared_expert_gate = KTCPULinear(
                layer.mlp.shared_expert_gate,
                dense_threads=self.cfg.dense_threads,
                stride=1,
                group_max_len=self.cfg.chunked_prefill_size,
                debug_name=f"layer{layer_idx}.shared_expert_gate",
                debug_finite=self.cfg.debug_finite,
            )
            print(f"[setup] layer={layer_idx} shared_expert=kt-kernel-mlp gate=kt-kernel-linear", flush=True)

    def _install_finite_debug_hooks(self) -> None:
        def make_hook(layer_idx: int):
            def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
                tensor = output[0] if isinstance(output, tuple) else output
                if isinstance(tensor, torch.Tensor) and not torch.isfinite(tensor).all():
                    finite = tensor[torch.isfinite(tensor)]
                    if finite.numel() > 0:
                        finite_min = float(finite.min().item())
                        finite_max = float(finite.max().item())
                    else:
                        finite_min = float("nan")
                        finite_max = float("nan")
                    raise RuntimeError(
                        f"non-finite hidden state after layer {layer_idx}: "
                        f"dtype={tensor.dtype} shape={tuple(tensor.shape)} "
                        f"finite_min={finite_min} finite_max={finite_max}"
                    )

            return hook

        for layer_idx, layer in enumerate(self.model.model.language_model.layers):
            layer.register_forward_hook(make_hook(layer_idx))
        print("[setup] finite debug hooks installed", flush=True)

    @property
    def device(self) -> torch.device:
        for parameter in self.model.parameters():
            if parameter.device.type != "meta":
                return parameter.device
        return torch.device("cuda:0")

    def prepare_prompt(self, prompt: str, *, thinking: bool) -> str:
        if not thinking:
            return prompt
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

    def forward_topk(self, prompt: str, topk: int = 5, *, thinking: bool = False) -> list[tuple[str, float]]:
        prompt_text = self.prepare_prompt(prompt, thinking=thinking)
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs, use_cache=True, return_dict=True)
        logits = outputs.logits[0, -1].float()
        values, indices = torch.topk(logits, k=topk)
        return [(self.tokenizer.decode([idx]), float(val)) for idx, val in zip(indices.tolist(), values.tolist())]

    def generate_text(self, prompt: str, *, thinking: bool = False, max_new_tokens: int | None = None) -> str:
        prompt_text = self.prepare_prompt(prompt, thinking=thinking)
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.cfg.max_new_tokens,
                do_sample=False,
            )
        new_tokens = generated[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=not thinking).strip()

    def benchmark_prefill_decode(
        self,
        prompt: str,
        *,
        decode_steps: int,
        thinking: bool = False,
        include_output_text: bool = False,
    ) -> dict[str, float | int | str]:
        prompt_text = self.prepare_prompt(prompt, thinking=thinking)
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[1])

        with torch.inference_mode():
            prefill_started = time.time()
            outputs = self.model(**inputs, use_cache=True, return_dict=True)
            torch.cuda.synchronize()
            prefill_elapsed = time.time() - prefill_started

            self.route_stats.reset()
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            attention_mask = inputs["attention_mask"]
            generated_tokens = [next_token]

            decode_started = time.time()
            for step in range(decode_steps):
                step_attention = torch.ones(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([attention_mask, step_attention], dim=1)
                outputs = self.model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values
                if step + 1 < decode_steps:
                    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                    generated_tokens.append(next_token)
            torch.cuda.synchronize()
            decode_elapsed = time.time() - decode_started

        result: dict[str, float | int | str] = {
            "prompt_tokens": prompt_tokens,
            "prefill_elapsed_s": prefill_elapsed,
            "prefill_token_per_s": prompt_tokens / prefill_elapsed if prefill_elapsed > 0 else float("nan"),
            "decode_steps": decode_steps,
            "decode_elapsed_s": decode_elapsed,
            "decode_token_per_s": decode_steps / decode_elapsed if decode_elapsed > 0 else float("nan"),
        }
        if include_output_text:
            result["output_text"] = self.tokenizer.decode(
                torch.cat(generated_tokens, dim=1)[0],
                skip_special_tokens=not thinking,
            ).strip()
        return result


def configure_torch_threads(num_threads: int, num_interop_threads: int) -> None:
    torch.set_num_threads(max(1, num_threads))
    torch.set_num_interop_threads(max(1, num_interop_threads))
    print(
        "[setup] "
        f"torch_num_threads={torch.get_num_threads()} "
        f"torch_num_interop_threads={torch.get_num_interop_threads()}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", default="/mnt/data/Qwen3.5-397B-A17B")
    parser.add_argument("--storage-root", default="/mnt/test/qwen35_397b_runtime_full_rust_fixed")
    parser.add_argument("--prompt", default="Answer with one short sentence: what is 2 plus 2?")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--decode-benchmark", action="store_true")
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--benchmark-print-output", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--cache-size-gib", type=float, default=48.0)
    parser.add_argument("--codebook-workers", type=int, default=1)
    parser.add_argument("--bitplane-workers", type=int, default=4)
    parser.add_argument("--compute-threads", type=int, default=8)
    parser.add_argument("--threadpool-count", type=int, default=1)
    parser.add_argument("--chunked-prefill-size", type=int, default=4096)
    parser.add_argument("--routed-backend", choices=("pagedmoe-python", "kt-wrapper"), default="pagedmoe-python")
    parser.add_argument("--kt-shared-expert", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dense-threads", type=int, default=8)
    parser.add_argument("--dense-stride", type=int, default=32)
    parser.add_argument("--debug-finite", action="store_true")
    parser.add_argument("--kt-attention", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the Qwen3.5 pagedmoe inference path")
    if args.kt_attention:
        raise SystemExit(
            "Qwen3.5 attention is not replaceable by an existing kt-kernel operator in this build. "
            "The model uses Qwen3_5MoeAttention/Qwen3_5MoeGatedDeltaNet, while kt-kernel exposes "
            "DeepSeek MLA/KV-cache operators with different semantics."
        )

    configure_torch_threads(args.torch_threads, args.torch_interop_threads)
    cfg = DirectRuntimeConfig(
        model_root=pathlib.Path(args.model_root),
        storage_root=pathlib.Path(args.storage_root),
        routed_backend=args.routed_backend,
        cache_size_gib=args.cache_size_gib,
        codebook_workers=args.codebook_workers,
        bitplane_workers=args.bitplane_workers,
        compute_threads=args.compute_threads,
        torch_threads=args.torch_threads,
        torch_interop_threads=args.torch_interop_threads,
        threadpool_count=args.threadpool_count,
        chunked_prefill_size=args.chunked_prefill_size,
        kt_shared_expert=args.kt_shared_expert,
        dense_threads=args.dense_threads,
        dense_stride=args.dense_stride,
        debug_finite=args.debug_finite,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"[setup] model_root={cfg.model_root}", flush=True)
    print(f"[setup] pagedmoe_storage_root={cfg.storage_root}", flush=True)
    print(
        "[setup] "
        f"routed_backend={cfg.routed_backend} "
        f"cache_size_gib={cfg.cache_size_gib} "
        f"codebook_workers={cfg.codebook_workers} "
        f"bitplane_workers={cfg.bitplane_workers} "
        f"compute_threads={cfg.compute_threads} "
        f"kt_shared_expert={cfg.kt_shared_expert} "
        f"dense_threads={cfg.dense_threads} "
        f"pagedmoe_target_precisions={PAGEDMOE_TARGET_PRECISIONS}",
        flush=True,
    )

    model = Qwen35KTPagedMoeModel(cfg)
    print("[setup] model ready", flush=True)

    if args.decode_benchmark:
        result = model.benchmark_prefill_decode(
            args.prompt,
            decode_steps=args.decode_steps,
            thinking=args.thinking,
            include_output_text=args.benchmark_print_output,
        )
        print(
            "[benchmark] "
            f"cache_size_gib={cfg.cache_size_gib} "
            f"prompt_tokens={result['prompt_tokens']} "
            f"prefill_elapsed_s={result['prefill_elapsed_s']:.4f} "
            f"prefill_token_per_s={result['prefill_token_per_s']:.4f} "
            f"decode_steps={result['decode_steps']} "
            f"decode_elapsed_s={result['decode_elapsed_s']:.4f} "
            f"decode_token_per_s={result['decode_token_per_s']:.4f}",
            flush=True,
        )
        if args.benchmark_print_output:
            print(f"[benchmark-output] {result['output_text']}", flush=True)
    elif args.forward_only:
        print("[forward-only]", model.forward_topk(args.prompt, thinking=args.thinking), flush=True)
    else:
        print("[generate]", model.generate_text(args.prompt, thinking=args.thinking), flush=True)

    print(f"[runtime] {model.route_stats.render()}", flush=True)


if __name__ == "__main__":
    main()
