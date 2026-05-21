# KT-Kernel Quickstart: SGLang + AMXINT8 / AMXINT4

This quickstart shows the shortest path for serving a MoE model with SGLang and KT-Kernel,
using Intel AMX kernels for CPU-side experts. The GPU keeps the base SGLang model weights,
while KT-Kernel loads converted CPU expert weights through `--kt-weight-path`.

Use `AMXINT8` as the accuracy-first default. Use `AMXINT4` when memory footprint and CPU
bandwidth matter more, and re-check accuracy for your model.

## 1. Check Hardware And Environment

AMX requires Intel Sapphire Rapids or newer, with AMX enabled by BIOS and OS:

```bash
lscpu | grep -i amx
```

Expected flags include:

```text
amx_bf16 amx_tile amx_int8
```

Install KTransformers / KT-Kernel together with the kvcache-ai SGLang fork:

```bash
git clone --recursive https://github.com/kvcache-ai/ktransformers.git
cd ktransformers

# Installs KT-Kernel and the SGLang fork with KT integration.
./install.sh
```

Do not use the upstream `sglang` package for these commands. The KT arguments below require
the KTransformers SGLang fork, or the `sglang-kt` package.

## 2. Choose Runtime Parameters

Set these once and reuse them in the conversion and launch commands:

```bash
export MODEL_PATH=/mnt/data/models/Qwen3-30B-A3B
export CPU_THREADS=64
export THREADPOOLS=2
export GPU_EXPERTS=32
export TP_SIZE=1
export PORT=30000
export SERVED_MODEL_NAME=qwen3-30b-a3b
```

Guidelines:

- `CPU_THREADS`: physical CPU cores, not hyperthreads.
- `THREADPOOLS`: usually the NUMA node count.
- `GPU_EXPERTS`: number of experts per MoE layer kept on GPU. Lower it if the server OOMs;
  raise it if GPU memory is available.
- `TP_SIZE`: SGLang tensor parallel size. Keep it `1` for a single GPU; set it to GPU count for a multi-GPU run.

Useful checks:

```bash
lscpu | grep -E '^CPU\(s\)|Thread\(s\) per core|NUMA node\(s\)'
numactl --hardware | grep available
```

## 3. Convert CPU Expert Weights

AMX backends need CPU-side expert weights converted by `kt-kernel/scripts/convert_cpu_weights.py`.

### AMXINT8 Weights

```bash
cd /path/to/ktransformers/kt-kernel

python scripts/convert_cpu_weights.py \
  --input-path "${MODEL_PATH}" \
  --input-type bf16 \
  --output "${MODEL_PATH}-AMXINT8" \
  --quant-method int8 \
  --cpuinfer-threads "${CPU_THREADS}" \
  --threadpool-count "${THREADPOOLS}"
```

### AMXINT4 Weights

```bash
cd /path/to/ktransformers/kt-kernel

python scripts/convert_cpu_weights.py \
  --input-path "${MODEL_PATH}" \
  --input-type bf16 \
  --output "${MODEL_PATH}-AMXINT4" \
  --quant-method int4 \
  --cpuinfer-threads "${CPU_THREADS}" \
  --threadpool-count "${THREADPOOLS}"
```

Set `--input-type` to match the source model weights. Supported values are `bf16`, `fp16`,
`fp8`, and `awq`; common Hugging Face BF16 models use `bf16`.

## 4. Launch SGLang With AMXINT8

```bash
python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --attention-backend triton \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 4096 \
  --max-running-requests 4 \
  --max-total-tokens 4096 \
  --enable-mixed-chunk \
  --tensor-parallel-size "${TP_SIZE}" \
  --disable-shared-experts-fusion \
  --kt-method AMXINT8 \
  --kt-weight-path "${MODEL_PATH}-AMXINT8" \
  --kt-cpuinfer "${CPU_THREADS}" \
  --kt-threadpool-count "${THREADPOOLS}" \
  --kt-num-gpu-experts "${GPU_EXPERTS}" \
  --kt-max-deferred-experts-per-token 2
```

## 5. Launch SGLang With AMXINT4

Only the converted weight directory and `--kt-method` differ:

```bash
python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --attention-backend triton \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 4096 \
  --max-running-requests 4 \
  --max-total-tokens 4096 \
  --enable-mixed-chunk \
  --tensor-parallel-size "${TP_SIZE}" \
  --disable-shared-experts-fusion \
  --kt-method AMXINT4 \
  --kt-weight-path "${MODEL_PATH}-AMXINT4" \
  --kt-cpuinfer "${CPU_THREADS}" \
  --kt-threadpool-count "${THREADPOOLS}" \
  --kt-num-gpu-experts "${GPU_EXPERTS}" \
  --kt-max-deferred-experts-per-token 2
```

## 6. Multi-GPU Notes

For a multi-GPU SGLang run, expose the GPUs in the order you want SGLang to see them, then set `TP_SIZE` accordingly:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx,GPU-yyyyyyyy-yyyy-yyyy-yyyy-yyyyyyyyyyyy
export TP_SIZE=2
```

If NCCL or custom all-reduce is unstable on the machine, add:

```bash
  --disable-custom-all-reduce
```

## 7. Smoke Test

Check the server:

```bash
curl -s http://127.0.0.1:${PORT}/health
```

Send an OpenAI-compatible request:

```bash
curl -s http://127.0.0.1:${PORT}/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"${SERVED_MODEL_NAME}\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"Write one sentence about heterogeneous MoE inference.\"}
    ],
    \"max_tokens\": 64
  }"
```

## Common Pitfalls

- `--kt-weight-path` for `AMXINT8` / `AMXINT4` must point to the converted CPU weight
  directory, not the original Hugging Face model directory.
- `--quant-method int8` pairs with `--kt-method AMXINT8`; `--quant-method int4` pairs with `--kt-method AMXINT4`.
- `--kt-cpuinfer` should be physical cores. Using hyperthread count often hurts performance.
- `--kt-threadpool-count` should match NUMA topology. If you bind explicit NUMA nodes, its
  length must match `--kt-threadpool-count`.
- `--kt-num-gpu-experts` is a memory/performance knob. If startup OOMs, lower it first.
- AMXINT4 can be materially less accurate for some models. Run a task-level evaluation before
  treating it as a drop-in replacement for AMXINT8.
