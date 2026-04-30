#!/usr/bin/env python3
"""
Launch and benchmark the full SGLang/kt-kernel inference path with PagedMoE.

This path keeps the serving/decode stack in SGLang + kt-kernel and replaces the
KT CPU MoE backend with the PAGEDMOE C ABI backend:

    sglang.launch_server
      -> SGLang Qwen3.5 model
      -> KTEPWrapperMethod
      -> KTMoEWrapper(method="PAGEDMOE")
      -> pagedmoe_runtime_execute_batch_sum_bf16()
"""

from __future__ import annotations

import argparse
import datetime
import json
import numbers
import os
import pathlib
import re
import signal
import site
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any


THIS_FILE = pathlib.Path(__file__).resolve()
KTRANSFORMERS_ROOT = THIS_FILE.parents[2]
PAGEDMOE_ROOT = KTRANSFORMERS_ROOT.parents[1]
SGLANG_PYTHON = KTRANSFORMERS_ROOT / "third_party" / "sglang" / "python"
KT_KERNEL_PYTHON = KTRANSFORMERS_ROOT / "kt-kernel" / "python"
PAGEDMOE_TARGET_RELEASE = PAGEDMOE_ROOT / "src" / "target" / "release"
LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
PAGEDMOE_STATS_PREFIX = "PagedMoe runtime stats:"
MCQ_LABELS = ("A", "B", "C", "D")
LOG_TS_PATTERN = r"\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?: [^\]]*)?\]"
PREFILL_LOG_RE = re.compile(
    LOG_TS_PATTERN
    + r".*Prefill batch(?: \[\d+\])?,\s+"
    + r"#new-seq:\s*(?P<new_seq>\d+),\s+"
    + r"#new-token:\s*(?P<new_token>\d+),\s+"
    + r"#cached-token:\s*(?P<cached_token>\d+),"
)
DECODE_LOG_RE = re.compile(
    LOG_TS_PATTERN
    + r".*Decode batch(?: \[\d+\])?,.*?"
    + r"#(?:full )?token:\s*(?P<decode_token>\d+),.*?"
    + r"gen throughput \(token/s\):\s*(?P<gen_throughput>[0-9.]+)"
)


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    python_paths = [str(SGLANG_PYTHON), str(KT_KERNEL_PYTHON)]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env["SGLANG_SRT_ONLY"] = "1"

    env["PAGEDMOE_CACHE_SIZE_GIB"] = str(args.cache_size_gib)
    env["PAGEDMOE_CODEBOOK_WORKERS"] = str(args.codebook_workers)
    env["PAGEDMOE_BITPLANE_WORKERS"] = str(args.bitplane_workers)
    env["PAGEDMOE_COMPUTE_THREADS"] = str(args.compute_threads)
    if args.pin_compute_workers:
        env["PAGEDMOE_PIN_COMPUTE_WORKERS"] = "1"

    ld_paths = [str(PAGEDMOE_TARGET_RELEASE), "/usr/local/cuda/lib64"]
    for site_dir in site.getsitepackages():
        nvidia_dir = pathlib.Path(site_dir) / "nvidia"
        if nvidia_dir.exists():
            ld_paths.extend(str(path) for path in nvidia_dir.glob("*/lib"))
    if env.get("LD_LIBRARY_PATH"):
        ld_paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_paths)
    env.setdefault("CUDA_HOME", "/usr/local/cuda")
    env.setdefault("CUDA_PATH", "/usr/local/cuda")
    env["PATH"] = f"/usr/local/cuda/bin{os.pathsep}{env.get('PATH', '')}"
    return env


def build_server_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.python,
        "-m",
        "sglang.launch_server",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model",
        str(args.model_root),
        "--kt-weight-path",
        str(args.storage_root),
        "--kt-cpuinfer",
        str(args.compute_threads),
        "--kt-threadpool-count",
        str(args.threadpool_count),
        "--kt-num-gpu-experts",
        str(args.gpu_experts),
        "--kt-method",
        "PAGEDMOE",
        "--kt-gpu-prefill-token-threshold",
        str(args.kt_gpu_prefill_threshold),
        "--attention-backend",
        args.attention_backend,
        "--trust-remote-code",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--chunked-prefill-size",
        str(args.chunked_prefill_size),
        "--max-running-requests",
        str(args.max_running_requests),
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--watchdog-timeout",
        str(args.watchdog_timeout),
        "--enable-mixed-chunk",
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--disable-shared-experts-fusion",
        "--skip-server-warmup",
    ]
    if args.disable_cuda_graph:
        cmd.append("--disable-cuda-graph")
    if args.random_seed is not None:
        cmd.extend(["--random-seed", str(args.random_seed)])
    if args.enable_p2p_check:
        cmd.append("--enable-p2p-check")
    if args.language_only:
        cmd.append("--language-only")
    if args.served_model_name:
        cmd.extend(["--served-model-name", args.served_model_name])
    if args.extra_sglang_args:
        cmd.extend(args.extra_sglang_args)
    return cmd


def http_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> Any:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with LOCAL_OPENER.open(req, timeout=timeout) as resp:
        body = resp.read()
    if not body:
        return None
    return json.loads(body.decode("utf-8"))


def wait_until_ready(base_url: str, proc: subprocess.Popen[Any] | None, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    ready_paths = ("/health", "/model_info", "/get_model_info")
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        for path in ready_paths:
            try:
                LOCAL_OPENER.open(f"{base_url}{path}", timeout=2).read()
                return
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code == 503:
                    continue
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        time.sleep(2)
    raise TimeoutError(f"server did not become ready within {timeout_s}s: {last_error}")


def load_tokenizer(args: argparse.Namespace) -> Any | None:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(args.model_root), trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] tokenizer load failed, using raw prompts: {exc}", flush=True)
        return None


def apply_chat_template(tokenizer: Any | None, text: str, thinking: bool) -> str:
    if tokenizer is None:
        return text
    try:
        messages = [{"role": "user", "content": text}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] tokenizer chat template failed, using raw prompt: {exc}", flush=True)
        return text


def request_generate(
    base_url: str,
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    ignore_eos: bool,
) -> dict[str, Any]:
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "skip_special_tokens": False,
            "ignore_eos": ignore_eos,
        },
    }
    try:
        return http_json("POST", f"{base_url}/generate", payload, timeout=3600)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"[warn] /generate failed with {exc.code}: {body}", flush=True)

    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
        "stream": False,
        "ignore_eos": ignore_eos,
    }
    return http_json("POST", f"{base_url}/v1/completions", payload, timeout=3600)


def extract_text_and_counts(response: Any, tokenizer: Any | None) -> tuple[str, int | None]:
    if isinstance(response, dict) and "text" in response:
        text = response.get("text") or ""
        meta = response.get("meta_info") or {}
        count = meta.get("completion_tokens") or meta.get("output_tokens")
        return text, count
    if isinstance(response, list) and response and isinstance(response[0], dict):
        return extract_text_and_counts(response[0], tokenizer)
    if isinstance(response, dict) and response.get("choices"):
        choice = response["choices"][0]
        text = choice.get("text") or choice.get("message", {}).get("content") or ""
        usage = response.get("usage") or {}
        return text, usage.get("completion_tokens")
    text = str(response)
    if tokenizer is not None:
        return text, len(tokenizer.encode(text, add_special_tokens=False))
    return text, None


def parse_server_decode_stats(log_file: pathlib.Path) -> dict[str, Any] | None:
    if not log_file.exists():
        return None

    pending_prefill: dict[str, Any] | None = None
    active_segment: dict[str, Any] | None = None
    last_segment: dict[str, Any] | None = None
    total_decode_log_lines = 0

    with log_file.open(encoding="utf-8", errors="replace") as log:
        for line in log:
            prefill_match = PREFILL_LOG_RE.search(line)
            if prefill_match:
                prefill_ts = datetime.datetime.strptime(
                    prefill_match.group("ts"), "%Y-%m-%d %H:%M:%S"
                )
                pending_prefill = {
                    "prefill_ts": prefill_ts,
                    "prefill_tokens": int(prefill_match.group("new_token"))
                    + int(prefill_match.group("cached_token")),
                }
                active_segment = None
                continue

            decode_match = DECODE_LOG_RE.search(line)
            if not decode_match:
                continue

            total_decode_log_lines += 1
            if pending_prefill is None:
                continue

            if active_segment is None:
                active_segment = {
                    **pending_prefill,
                    "decode_log_lines": 0,
                }

            active_segment["decode_log_lines"] += 1
            active_segment["last_decode_ts"] = datetime.datetime.strptime(
                decode_match.group("ts"), "%Y-%m-%d %H:%M:%S"
            )
            active_segment["last_decode_tokens"] = int(decode_match.group("decode_token"))
            last_segment = active_segment

    if last_segment is None:
        return None

    decode_tokens = last_segment["last_decode_tokens"] - last_segment["prefill_tokens"]
    decode_elapsed_s = (
        last_segment["last_decode_ts"] - last_segment["prefill_ts"]
    ).total_seconds()
    if decode_tokens <= 0 or decode_elapsed_s <= 0:
        return None

    return {
        **last_segment,
        "total_decode_log_lines": total_decode_log_lines,
        "decode_tokens": decode_tokens,
        "decode_elapsed_s": decode_elapsed_s,
        "decode_token_per_s": decode_tokens / decode_elapsed_s,
    }


def print_decode_stats_from_log(log_file: pathlib.Path) -> None:
    stats = parse_server_decode_stats(log_file)
    if stats is None:
        return

    print(
        "[benchmark] "
        f"decode_log_lines={stats['decode_log_lines']} "
        f"decode_log_total_lines={stats['total_decode_log_lines']} "
        f"decode_log_prefill_ts={stats['prefill_ts'].isoformat()} "
        f"decode_log_last_ts={stats['last_decode_ts'].isoformat()} "
        f"decode_log_prefill_tokens={stats['prefill_tokens']} "
        f"decode_log_last_tokens={stats['last_decode_tokens']} "
        f"decode_log_tokens_used={stats['decode_tokens']} "
        f"decode_log_elapsed_s={stats['decode_elapsed_s']:.4f} "
        f"decode_token_per_s_from_log={stats['decode_token_per_s']:.4f}",
        flush=True,
    )


def run_benchmark(args: argparse.Namespace) -> None:
    base_url = f"http://{args.host}:{args.port}"
    tokenizer = load_tokenizer(args)
    prompt = apply_chat_template(tokenizer, args.prompt, args.thinking)
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False)) if tokenizer is not None else None
    print(f"[benchmark] prompt_tokens={prompt_tokens if prompt_tokens is not None else 'unknown'}", flush=True)
    start = time.perf_counter()
    response = request_generate(
        base_url,
        args.served_model_name or args.model_root.name,
        prompt,
        args.decode_steps,
        args.ignore_eos,
    )
    elapsed = time.perf_counter() - start
    text, completion_tokens = extract_text_and_counts(response, tokenizer)
    if completion_tokens is None:
        completion_tokens = args.decode_steps
    print(
        "[benchmark] "
        f"cache_size_gib={args.cache_size_gib} "
        f"compute_threads={args.compute_threads} "
        f"decode_steps={args.decode_steps} "
        f"elapsed_s={elapsed:.4f} "
        f"completion_tokens={completion_tokens} "
        f"completion_token_per_s={completion_tokens / elapsed:.4f}",
        flush=True,
    )
    if args.print_output:
        print("[output-begin]", flush=True)
        print(text, flush=True)
        print("[output-end]", flush=True)


def build_mmlu_prompt(question: str, choices: list[str]) -> str:
    return (
        "Answer the following multiple choice question. "
        "Respond with only one letter: A, B, C, or D.\n\n"
        f"Question: {question}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n\n"
        "Answer:"
    )


def extract_choice(text: str) -> str:
    match = re.search(r"\b([ABCD])\b", text.upper())
    return match.group(1) if match else ""


def run_mmlu(args: argparse.Namespace) -> None:
    from datasets import load_dataset

    base_url = f"http://{args.host}:{args.port}"
    tokenizer = load_tokenizer(args)
    ds = load_dataset("cais/mmlu", args.mmlu_subject, split=args.mmlu_split)
    limit = len(ds) if args.mmlu_limit <= 0 else min(args.mmlu_limit, len(ds))
    correct = 0
    start = time.perf_counter()
    for idx in range(limit):
        row = ds[idx]
        prompt = apply_chat_template(
            tokenizer,
            build_mmlu_prompt(str(row["question"]), list(row["choices"])),
            thinking=False,
        )
        response = request_generate(
            base_url,
            args.served_model_name or args.model_root.name,
            prompt,
            args.mmlu_max_new_tokens,
            ignore_eos=False,
        )
        text, _ = extract_text_and_counts(response, tokenizer)
        pred = extract_choice(text)
        answer = row["answer"]
        gold = MCQ_LABELS[int(answer)] if isinstance(answer, numbers.Integral) else str(answer).strip().upper()
        ok = pred == gold
        correct += int(ok)
        print(
            f"[mmlu] idx={idx} subject={args.mmlu_subject} pred={pred!r} gold={gold!r} ok={ok} text={text!r}",
            flush=True,
        )
    elapsed = time.perf_counter() - start
    print(
        f"[mmlu] subject={args.mmlu_subject} split={args.mmlu_split} "
        f"limit={limit} accuracy={correct}/{limit} elapsed_s={elapsed:.2f}",
        flush=True,
    )


def print_pagedmoe_stats_from_log(log_file: pathlib.Path) -> None:
    if not log_file.exists():
        return
    stats_line = None
    for line in log_file.read_text(errors="replace").splitlines():
        if PAGEDMOE_STATS_PREFIX in line:
            stats_line = line
    if stats_line is not None:
        print("[pagedmoe]", stats_line.split(PAGEDMOE_STATS_PREFIX, 1)[1].strip(), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model-root", type=pathlib.Path, default=pathlib.Path("/mnt/data/Qwen3.5-397B-A17B"))
    parser.add_argument(
        "--storage-root",
        type=pathlib.Path,
        default=pathlib.Path("/mnt/test/qwen35_397b_runtime_full_rust_fixed"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--served-model-name", default="qwen35-pagedmoe-kt")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-experts", type=int, default=0)
    parser.add_argument("--compute-threads", type=int, default=8)
    parser.add_argument("--threadpool-count", type=int, default=1)
    parser.add_argument("--cache-size-gib", type=float, default=48.0)
    parser.add_argument("--codebook-workers", type=int, default=1)
    parser.add_argument("--bitplane-workers", type=int, default=4)
    parser.add_argument("--pin-compute-workers", action="store_true")
    parser.add_argument("--chunked-prefill-size", type=int, default=4096)
    parser.add_argument("--kt-gpu-prefill-threshold", type=int, default=1000000000)
    parser.add_argument("--attention-backend", default="triton")
    parser.add_argument("--max-running-requests", type=int, default=1)
    parser.add_argument("--max-total-tokens", type=int, default=4096)
    parser.add_argument("--mem-fraction-static", type=float, default=0.90)
    parser.add_argument("--watchdog-timeout", type=int, default=3000)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--enable-p2p-check", action="store_true")
    parser.add_argument("--language-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt", default="Answer with one short sentence: what is 2 plus 2?")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--decode-steps", type=int, default=50)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mmlu-subject", default=None)
    parser.add_argument("--mmlu-split", default="test")
    parser.add_argument("--mmlu-limit", type=int, default=0)
    parser.add_argument("--mmlu-max-new-tokens", type=int, default=4)
    parser.add_argument("--print-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--startup-timeout", type=float, default=1800.0)
    parser.add_argument(
        "--shutdown-signal",
        choices=("SIGTERM", "SIGINT"),
        default="SIGTERM",
        help="Signal sent to a launched SGLang server after the benchmark finishes.",
    )
    parser.add_argument("--shutdown-timeout", type=float, default=60.0)
    parser.add_argument("--log-file", type=pathlib.Path, default=pathlib.Path("/tmp/qwen35_pagedmoe_kt_server.log"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--server-only", action="store_true")
    parser.add_argument("--no-launch", action="store_true", help="Benchmark an already running server")
    parser.add_argument("extra_sglang_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.extra_sglang_args and args.extra_sglang_args[0] == "--":
        args.extra_sglang_args = args.extra_sglang_args[1:]
    cmd = build_server_cmd(args)
    env = build_env(args)
    if args.dry_run:
        print("[env] SGLANG_SRT_ONLY=1", flush=True)
        for key in (
            "PAGEDMOE_CACHE_SIZE_GIB",
            "PAGEDMOE_CODEBOOK_WORKERS",
            "PAGEDMOE_BITPLANE_WORKERS",
            "PAGEDMOE_COMPUTE_THREADS",
        ):
            print(f"[env] {key}={env[key]}", flush=True)
        print("[cmd]", " ".join(cmd), flush=True)
        return

    proc = None
    log_fp = None
    try:
        if not args.no_launch:
            args.log_file.parent.mkdir(parents=True, exist_ok=True)
            log_fp = args.log_file.open("wb")
            print(f"[server] log_file={args.log_file}", flush=True)
            print("[server] launching:", " ".join(cmd), flush=True)
            proc = subprocess.Popen(cmd, env=env, stdout=log_fp, stderr=subprocess.STDOUT)
        wait_until_ready(f"http://{args.host}:{args.port}", proc, args.startup_timeout)
        print("[server] ready", flush=True)
        if not args.server_only:
            if args.mmlu_subject:
                run_mmlu(args)
            else:
                run_benchmark(args)
    finally:
        if proc is not None and proc.poll() is None:
            shutdown_signal = getattr(signal, args.shutdown_signal)
            print(f"[server] stopping with {args.shutdown_signal}", flush=True)
            proc.send_signal(shutdown_signal)
            try:
                proc.wait(timeout=args.shutdown_timeout)
            except subprocess.TimeoutExpired:
                print("[server] graceful stop timed out; killing server", flush=True)
                proc.kill()
        if log_fp is not None:
            log_fp.close()
        if proc is not None and not args.server_only:
            print_decode_stats_from_log(args.log_file)
            print_pagedmoe_stats_from_log(args.log_file)


if __name__ == "__main__":
    main()
